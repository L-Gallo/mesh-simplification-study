#!/usr/bin/env python3
"""
Mesh Simplification Comparison Tool
====================================
A unified testing framework comparing fast-simplification (PyVista) and Open3D
mesh simplification libraries for academic research.

Author: Research Tool for Master's Thesis
Purpose: Compare traditional geometric mesh simplification methods for game industry
Version: 1.0.0

Installation:
    pip install pyvista fast-simplification open3d psutil numpy

Usage:
    python mesh_simplifier.py --input mesh.ply --method fast-simplification
    python mesh_simplifier.py --input mesh.ply --method open3d
    python mesh_simplifier.py --input mesh.ply --method both --output-dir ./results
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import psutil

# Conditional imports with error handling
try:
    import pyvista as pv
    import fast_simplification
    FAST_SIMPLIFICATION_AVAILABLE = True
except ImportError as e:
    FAST_SIMPLIFICATION_AVAILABLE = False
    FAST_SIMPLIFICATION_ERROR = str(e)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError as e:
    OPEN3D_AVAILABLE = False
    OPEN3D_ERROR = str(e)

try:
    import meshoptimizer
    MESHOPTIMIZER_AVAILABLE = True
except ImportError as e:
    MESHOPTIMIZER_AVAILABLE = False
    MESHOPTIMIZER_ERROR = str(e)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class SystemInfo:
    """System configuration for reproducibility."""
    python_version: str
    platform: str
    processor: str
    total_ram_gb: float
    cpu_count: int
    timestamp: str
    pyvista_version: Optional[str] = None
    fast_simplification_version: Optional[str] = None
    open3d_version: Optional[str] = None
    meshoptimizer_version: Optional[str] = None


@dataclass
class MeshMetrics:
    """Mesh statistics before and after simplification."""
    vertex_count: int
    face_count: int
    file_size_bytes: Optional[int] = None
    bounding_box_diagonal: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance profiling results."""
    execution_time_seconds: float
    execution_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_mb: float


@dataclass
class SimplificationResult:
    """Complete result of a single simplification operation."""
    method: str
    reduction_level: str  # "75%", "50%", "25%"
    target_reduction_ratio: float
    actual_reduction_ratio: float
    input_mesh_path: str
    output_mesh_path: str
    input_metrics: MeshMetrics
    output_metrics: MeshMetrics
    performance: PerformanceMetrics
    success: bool
    error_message: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report for all operations."""
    system_info: SystemInfo
    results: List[SimplificationResult]
    total_execution_time_seconds: float
    generated_at: str


# =============================================================================
# Memory Profiler
# =============================================================================

class MemoryProfiler:
    """
    Memory profiler combining tracemalloc (Python allocations) and psutil 
    (total process memory including C++ libraries like VTK and Open3D).
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._started = False
        self.start_rss: float = 0
        self.start_traced: float = 0
    
    def start(self) -> float:
        """Start memory profiling. Returns current RSS in MB."""
        tracemalloc.start()
        self._started = True
        self.start_rss = self.process.memory_info().rss / (1024 * 1024)
        self.start_traced, _ = tracemalloc.get_traced_memory()
        return self.start_rss
    
    def stop(self) -> Tuple[float, float, float]:
        """
        Stop memory profiling.
        
        Returns:
            Tuple of (memory_after_mb, memory_delta_mb, peak_memory_mb)
        """
        if not self._started:
            return 0.0, 0.0, 0.0
        
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._started = False
        
        end_rss = self.process.memory_info().rss / (1024 * 1024)
        delta_rss = end_rss - self.start_rss
        peak_mb = peak_traced / (1024 * 1024)
        
        return end_rss, delta_rss, peak_mb


# =============================================================================
# Mesh Simplifier Interface
# =============================================================================

class MeshSimplifier:
    """
    Unified interface for mesh simplification using either fast-simplification
    (PyVista) or Open3D libraries.
    
    Attributes:
        method: Either "fast-simplification" or "open3d"
        output_dir: Directory for saving simplified meshes
        logger: Logger instance for operation logging
    
    Example:
        simplifier = MeshSimplifier(method="fast-simplification", output_dir="./output")
        results = simplifier.run_benchmark("model.ply")
    """
    
    # Reduction levels: (level_name, fraction_to_keep)
    REDUCTION_LEVELS = [
        ("75%", 0.25),  # 75% reduction = keep 25% of triangles
        ("50%", 0.50),  # 50% reduction = keep 50% of triangles
        ("25%", 0.75),  # 25% reduction = keep 75% of triangles
    ]
    
    SUPPORTED_FORMATS = {'.ply', '.stl', '.obj', '.off', '.vtk'}
    
    def __init__(
        self,
        method: Literal["fast-simplification", "open3d", "meshoptimizer", "cgal", "all"],
        output_dir: Union[str, Path] = "./simplified_meshes",
        aggressiveness: int = 7,  # fast-simplification parameter (0-10)
        boundary_weight: float = 1.0,  # Open3D parameter
        cgal_executable: Union[str, Path] = "./cgal_simplify",  # Path to CGAL executable
        log_level: int = logging.INFO
    ):
        """
        Initialize the mesh simplifier.
        
        Args:
            method: Simplification library to use
            output_dir: Directory for output files
            aggressiveness: fast-simplification aggressiveness (0-10, default 7)
            boundary_weight: Open3D boundary preservation weight (default 1.0)
            cgal_executable: Path to CGAL simplification executable
            log_level: Logging verbosity level
        """
        self.method = method
        self.output_dir = Path(output_dir)
        self.aggressiveness = aggressiveness
        self.boundary_weight = boundary_weight
        self.cgal_executable = Path(cgal_executable)
        
        # Setup logging
        self.logger = logging.getLogger("MeshSimplifier")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.logger.addHandler(handler)
        
        # Validate library availability
        self._validate_libraries()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized MeshSimplifier with method='{method}'")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def _validate_libraries(self) -> None:
        """Validate that required libraries are available."""
        if self.method in ["fast-simplification", "all"]:
            if not FAST_SIMPLIFICATION_AVAILABLE:
                raise ImportError(
                    f"fast-simplification not available: {FAST_SIMPLIFICATION_ERROR}\n"
                    "Install with: pip install pyvista fast-simplification"
                )
        
        if self.method in ["open3d", "all"]:
            if not OPEN3D_AVAILABLE:
                raise ImportError(
                    f"Open3D not available: {OPEN3D_ERROR}\n"
                    "Install with: pip install open3d"
                )
        
        if self.method in ["meshoptimizer", "all"]:
            if not MESHOPTIMIZER_AVAILABLE:
                raise ImportError(
                    f"meshoptimizer not available: {MESHOPTIMIZER_ERROR}\n"
                    "Install with: pip install meshoptimizer"
                )
        
        if self.method in ["cgal", "all"]:
            if not self.cgal_executable.exists():
                raise FileNotFoundError(
                    f"CGAL executable not found: {self.cgal_executable}\n"
                    "Please compile cgal_simplify and provide the path via --cgal-executable"
                )
    
    def _get_system_info(self) -> SystemInfo:
        """Collect system information for reproducibility."""
        info = SystemInfo(
            python_version=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor() or "Unknown",
            total_ram_gb=psutil.virtual_memory().total / (1024 ** 3),
            cpu_count=os.cpu_count() or 1,
            timestamp=datetime.now().isoformat()
        )
        
        if FAST_SIMPLIFICATION_AVAILABLE:
            info.pyvista_version = pv.__version__
            info.fast_simplification_version = getattr(
                fast_simplification, '__version__', 'unknown'
            )
        
        if OPEN3D_AVAILABLE:
            info.open3d_version = o3d.__version__
        
        if MESHOPTIMIZER_AVAILABLE:
            info.meshoptimizer_version = getattr(
                meshoptimizer, '__version__', 'unknown'
            )
        
        return info
    
    def _get_mesh_metrics_pyvista(self, mesh: "pv.PolyData", filepath: Optional[Path] = None) -> MeshMetrics:
        """Extract metrics from a PyVista mesh."""
        file_size = filepath.stat().st_size if filepath and filepath.exists() else None
        
        # Calculate bounding box diagonal
        bounds = mesh.bounds
        diagonal = np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
        
        # Use n_cells instead of deprecated n_faces (PyVista >= 0.43)
        face_count = mesh.n_cells if hasattr(mesh, 'n_cells') else mesh.n_faces
        
        return MeshMetrics(
            vertex_count=mesh.n_points,
            face_count=face_count,
            file_size_bytes=file_size,
            bounding_box_diagonal=float(diagonal)
        )
    
    def _get_mesh_metrics_open3d(self, mesh: "o3d.geometry.TriangleMesh", filepath: Optional[Path] = None) -> MeshMetrics:
        """Extract metrics from an Open3D mesh."""
        file_size = filepath.stat().st_size if filepath and filepath.exists() else None
        
        # Calculate bounding box diagonal
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        diagonal = np.sqrt(extent[0]**2 + extent[1]**2 + extent[2]**2)
        
        return MeshMetrics(
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.triangles),
            file_size_bytes=file_size,
            bounding_box_diagonal=float(diagonal)
        )
    
    def _generate_output_filename(
        self,
        input_path: Path,
        method: str,
        reduction_level: str
    ) -> Path:
        """Generate descriptive output filename."""
        stem = input_path.stem
        suffix = input_path.suffix
        
        # Map method names to short codes
        method_short_map = {
            "fast-simplification": "fs",
            "open3d": "o3d",
            "meshoptimizer": "meshopt",
            "cgal": "cgal"
        }
        method_short = method_short_map.get(method, method)
        
        reduction_clean = reduction_level.replace("%", "pct")
        
        filename = f"{stem}_{method_short}_reduction{reduction_clean}{suffix}"
        return self.output_dir / filename
    
    # =========================================================================
    # Fast-Simplification Methods
    # =========================================================================
    
    def _simplify_fast_simplification(
        self,
        input_path: Path,
        target_keep_ratio: float,
        reduction_level: str
    ) -> SimplificationResult:
        """
        Simplify mesh using fast-simplification library.
        
        Args:
            input_path: Path to input mesh file
            target_keep_ratio: Fraction of triangles to KEEP (e.g., 0.25 keeps 25%)
            reduction_level: Human-readable reduction level ("75%", "50%", "25%")
        
        Returns:
            SimplificationResult with all metrics
        """
        output_path = self._generate_output_filename(
            input_path, "fast-simplification", reduction_level
        )
        
        self.logger.info(f"[fast-simplification] Starting {reduction_level} reduction...")
        
        profiler = MemoryProfiler()
        
        try:
            # Load mesh
            self.logger.debug(f"Loading mesh: {input_path}")
            mesh = pv.read(str(input_path))
            input_metrics = self._get_mesh_metrics_pyvista(mesh, input_path)
            
            self.logger.info(
                f"  Input: {input_metrics.vertex_count:,} vertices, "
                f"{input_metrics.face_count:,} faces"
            )
            
            # Calculate target_reduction for fast-simplification
            # fast-simplification: target_reduction = fraction to REMOVE
            target_reduction = 1.0 - target_keep_ratio
            
            # Profile the simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
            # Perform simplification
            simplified = fast_simplification.simplify_mesh(
                mesh,
                target_reduction=target_reduction,
                agg=self.aggressiveness,
                verbose=False
            )
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            
            execution_time = end_time - start_time
            
            # Save simplified mesh
            self.logger.debug(f"Saving to: {output_path}")
            simplified.save(str(output_path))
            
            output_metrics = self._get_mesh_metrics_pyvista(simplified, output_path)
            
            # Calculate actual reduction ratio
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            self.logger.info(
                f"  Output: {output_metrics.vertex_count:,} vertices, "
                f"{output_metrics.face_count:,} faces"
            )
            self.logger.info(
                f"  Time: {execution_time*1000:.2f} ms | "
                f"Memory delta: {memory_delta:+.2f} MB"
            )
            
            return SimplificationResult(
                method="fast-simplification",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=input_metrics,
                output_metrics=output_metrics,
                performance=PerformanceMetrics(
                    execution_time_seconds=execution_time,
                    execution_time_ms=execution_time * 1000,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    memory_delta_mb=memory_delta,
                    peak_memory_mb=peak_memory
                ),
                success=True,
                additional_params={"aggressiveness": self.aggressiveness}
            )
            
        except Exception as e:
            self.logger.error(f"  FAILED: {str(e)}")
            return SimplificationResult(
                method="fast-simplification",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                error_message=str(e)
            )
    
    # =========================================================================
    # Open3D Methods
    # =========================================================================
    
    def _simplify_open3d(
        self,
        input_path: Path,
        target_keep_ratio: float,
        reduction_level: str
    ) -> SimplificationResult:
        """
        Simplify mesh using Open3D library.
        
        Args:
            input_path: Path to input mesh file
            target_keep_ratio: Fraction of triangles to KEEP (e.g., 0.25 keeps 25%)
            reduction_level: Human-readable reduction level ("75%", "50%", "25%")
        
        Returns:
            SimplificationResult with all metrics
        """
        output_path = self._generate_output_filename(
            input_path, "open3d", reduction_level
        )
        
        self.logger.info(f"[Open3D] Starting {reduction_level} reduction...")
        
        profiler = MemoryProfiler()
        
        try:
            # Load mesh
            self.logger.debug(f"Loading mesh: {input_path}")
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            
            # Preprocessing (recommended for Open3D)
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            
            input_metrics = self._get_mesh_metrics_open3d(mesh, input_path)
            
            self.logger.info(
                f"  Input: {input_metrics.vertex_count:,} vertices, "
                f"{input_metrics.face_count:,} faces"
            )
            
            # Calculate target triangle count
            target_triangles = int(input_metrics.face_count * target_keep_ratio)
            target_triangles = max(target_triangles, 4)  # Minimum valid mesh
            
            # Profile the simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
            # Perform simplification using quadric decimation
            simplified = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles,
                maximum_error=float('inf'),
                boundary_weight=self.boundary_weight
            )
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            
            execution_time = end_time - start_time
            
            # Compute normals for proper visualization
            simplified.compute_vertex_normals()
            
            # Save simplified mesh
            self.logger.debug(f"Saving to: {output_path}")
            o3d.io.write_triangle_mesh(str(output_path), simplified)
            
            output_metrics = self._get_mesh_metrics_open3d(simplified, output_path)
            
            # Calculate actual reduction ratio
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            self.logger.info(
                f"  Output: {output_metrics.vertex_count:,} vertices, "
                f"{output_metrics.face_count:,} faces"
            )
            self.logger.info(
                f"  Time: {execution_time*1000:.2f} ms | "
                f"Memory delta: {memory_delta:+.2f} MB"
            )
            
            return SimplificationResult(
                method="open3d",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=input_metrics,
                output_metrics=output_metrics,
                performance=PerformanceMetrics(
                    execution_time_seconds=execution_time,
                    execution_time_ms=execution_time * 1000,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    memory_delta_mb=memory_delta,
                    peak_memory_mb=peak_memory
                ),
                success=True,
                additional_params={
                    "boundary_weight": self.boundary_weight,
                    "target_triangles": target_triangles
                }
            )
            
        except Exception as e:
            self.logger.error(f"  FAILED: {str(e)}")
            return SimplificationResult(
                method="open3d",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                error_message=str(e)
            )
    
    # =========================================================================
    # meshoptimizer Methods
    # =========================================================================
    
    def _simplify_meshoptimizer(
        self,
        input_path: Path,
        target_keep_ratio: float,
        reduction_level: str
    ) -> SimplificationResult:
        """
        Simplify mesh using meshoptimizer library.
        
        Args:
            input_path: Path to input mesh file
            target_keep_ratio: Fraction of triangles to KEEP (e.g., 0.25 keeps 25%)
            reduction_level: Human-readable reduction level ("75%", "50%", "25%")
        
        Returns:
            SimplificationResult with all metrics
        """
        output_path = self._generate_output_filename(
            input_path, "meshoptimizer", reduction_level
        )
        
        self.logger.info(f"[meshoptimizer] Starting {reduction_level} reduction...")
        
        profiler = MemoryProfiler()
        
        try:
            # Load mesh using PyVista
            self.logger.debug(f"Loading mesh: {input_path}")
            mesh = pv.read(str(input_path))
            input_metrics = self._get_mesh_metrics_pyvista(mesh, input_path)
            
            self.logger.info(
                f"  Input: {input_metrics.vertex_count:,} vertices, "
                f"{input_metrics.face_count:,} faces"
            )
            
            # Convert to numpy arrays (meshoptimizer format)
            vertices = np.array(mesh.points, dtype=np.float32)
            
            # Extract indices from PyVista format (faces array has size prefix)
            faces_array = mesh.faces
            # PyVista format: [n, v0, v1, v2, n, v0, v1, v2, ...]
            # Reshape and extract just the vertex indices
            n_faces = input_metrics.face_count
            indices = faces_array.reshape(n_faces, 4)[:, 1:4].astype(np.uint32)
            indices_flat = indices.flatten()
            
            # Calculate target triangle count
            target_triangles = int(input_metrics.face_count * target_keep_ratio)
            target_triangles = max(target_triangles, 4)  # Minimum valid mesh
            target_index_count = target_triangles * 3
            
            # Profile the simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
            # Perform simplification
            # Pre-allocate destination buffer (worst case: same size as input)
            destination = np.zeros(len(indices_flat), dtype=np.uint32)
            target_error = 0.01  # 1% error threshold
            
            # meshoptimizer.simplify writes to destination and returns count
            result_count = meshoptimizer.simplify(
                destination=destination,
                indices=indices_flat,
                vertex_positions=vertices,
                target_index_count=target_index_count,
                target_error=target_error,
                options=0
            )
            
            # Extract the actual simplified indices
            simplified_indices = destination[:result_count]
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            
            execution_time = end_time - start_time
            
            # Reconstruct mesh from simplified indices
            # meshoptimizer returns flat array, reshape to triangles
            n_simplified_triangles = len(simplified_indices) // 3
            simplified_faces = simplified_indices.reshape(n_simplified_triangles, 3)
            
            # Find used vertices
            used_vertices = np.unique(simplified_indices)
            new_vertices = vertices[used_vertices]
            
            # Remap indices to new compact vertex array
            vertex_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            remapped_indices = np.array([vertex_remap[idx] for idx in simplified_indices])
            remapped_faces = remapped_indices.reshape(n_simplified_triangles, 3)
            
            # Create PyVista mesh for saving
            # PyVista format: [n, v0, v1, v2, n, v0, v1, v2, ...]
            faces_with_size = np.column_stack([
                np.full(n_simplified_triangles, 3),
                remapped_faces
            ]).flatten()
            
            simplified = pv.PolyData(new_vertices, faces_with_size)
            
            # Save simplified mesh
            self.logger.debug(f"Saving to: {output_path}")
            simplified.save(str(output_path))
            
            output_metrics = self._get_mesh_metrics_pyvista(simplified, output_path)
            
            # Calculate actual reduction ratio
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            self.logger.info(
                f"  Output: {output_metrics.vertex_count:,} vertices, "
                f"{output_metrics.face_count:,} faces"
            )
            self.logger.info(
                f"  Time: {execution_time*1000:.2f} ms | "
                f"Memory delta: {memory_delta:+.2f} MB"
            )
            
            return SimplificationResult(
                method="meshoptimizer",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=input_metrics,
                output_metrics=output_metrics,
                performance=PerformanceMetrics(
                    execution_time_seconds=execution_time,
                    execution_time_ms=execution_time * 1000,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    memory_delta_mb=memory_delta,
                    peak_memory_mb=peak_memory
                ),
                success=True,
                additional_params={
                    "target_error": target_error,
                    "target_triangles": target_triangles
                }
            )
            
        except Exception as e:
            self.logger.error(f"  FAILED: {str(e)}")
            return SimplificationResult(
                method="meshoptimizer",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                error_message=str(e)
            )
    
    # =========================================================================
    # CGAL Methods (via C++ subprocess)
    # =========================================================================
    
    def _simplify_cgal(
        self,
        input_path: Path,
        target_keep_ratio: float,
        reduction_level: str
    ) -> SimplificationResult:
        """
        Simplify mesh using CGAL library (via C++ executable).
        
        Args:
            input_path: Path to input mesh file
            target_keep_ratio: Fraction of triangles to KEEP (e.g., 0.25 keeps 25%)
            reduction_level: Human-readable reduction level ("75%", "50%", "25%")
        
        Returns:
            SimplificationResult with all metrics
        """
        output_path = self._generate_output_filename(
            input_path, "cgal", reduction_level
        )
        
        stats_path = self.output_dir / f"cgal_stats_{reduction_level}.json"
        
        self.logger.info(f"[CGAL] Starting {reduction_level} reduction...")
        
        profiler = MemoryProfiler()
        
        try:
            # Get input mesh metrics using PyVista
            mesh = pv.read(str(input_path))
            input_metrics = self._get_mesh_metrics_pyvista(mesh, input_path)
            
            self.logger.info(
                f"  Input: {input_metrics.vertex_count:,} vertices, "
                f"{input_metrics.face_count:,} faces"
            )
            
            # Profile the simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
            # Call CGAL executable
            cmd = [
                str(self.cgal_executable),
                str(input_path),
                str(output_path),
                str(target_keep_ratio),
                str(stats_path)
            ]
            
            self.logger.debug(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            
            execution_time = end_time - start_time
            
            # Check if execution succeeded
            if result.returncode != 0:
                error_msg = f"CGAL process failed with code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                raise RuntimeError(error_msg)
            
            # Read statistics from JSON if available
            cgal_stats = {}
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    cgal_stats = json.load(f)
                stats_path.unlink()  # Clean up temp file
            
            # Get output mesh metrics
            if not output_path.exists():
                raise FileNotFoundError(f"CGAL did not create output file: {output_path}")
            
            simplified = pv.read(str(output_path))
            output_metrics = self._get_mesh_metrics_pyvista(simplified, output_path)
            
            # Calculate actual reduction ratio
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            self.logger.info(
                f"  Output: {output_metrics.vertex_count:,} vertices, "
                f"{output_metrics.face_count:,} faces"
            )
            self.logger.info(
                f"  Time: {execution_time*1000:.2f} ms | "
                f"Memory delta: {memory_delta:+.2f} MB"
            )
            
            return SimplificationResult(
                method="cgal",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=input_metrics,
                output_metrics=output_metrics,
                performance=PerformanceMetrics(
                    execution_time_seconds=execution_time,
                    execution_time_ms=execution_time * 1000,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    memory_delta_mb=memory_delta,
                    peak_memory_mb=peak_memory
                ),
                success=True,
                additional_params={
                    "cgal_execution_time_ms": cgal_stats.get("execution_time_ms"),
                    "lindstrom_turk_placement": True
                }
            )
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"  FAILED: CGAL process timed out after 300 seconds")
            return SimplificationResult(
                method="cgal",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                error_message="Process timeout"
            )
        except Exception as e:
            self.logger.error(f"  FAILED: {str(e)}")
            return SimplificationResult(
                method="cgal",
                reduction_level=reduction_level,
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                error_message=str(e)
            )
    
    # =========================================================================
    # Main Benchmark Runner
    # =========================================================================
    
    def run_benchmark(self, input_path: Union[str, Path]) -> BenchmarkReport:
        """
        Run complete benchmark on a mesh file.
        
        Generates simplified versions at 75%, 50%, and 25% reduction levels
        using the configured method(s).
        
        Args:
            input_path: Path to input mesh file
        
        Returns:
            BenchmarkReport with all results and system info
        """
        input_path = Path(input_path)
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input mesh not found: {input_path}")
        
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{input_path.suffix}'. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        self.logger.info("=" * 60)
        self.logger.info("MESH SIMPLIFICATION BENCHMARK")
        self.logger.info("=" * 60)
        self.logger.info(f"Input file: {input_path}")
        self.logger.info(f"Method(s): {self.method}")
        self.logger.info("")
        
        total_start = time.perf_counter()
        results: List[SimplificationResult] = []
        
        # Determine which methods to run
        methods_to_run = []
        if self.method in ["fast-simplification", "all"]:
            methods_to_run.append("fast-simplification")
        if self.method in ["open3d", "all"]:
            methods_to_run.append("open3d")
        if self.method in ["meshoptimizer", "all"]:
            methods_to_run.append("meshoptimizer")
        if self.method in ["cgal", "all"]:
            methods_to_run.append("cgal")
        
        # Run all combinations
        for method in methods_to_run:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running {method.upper()}")
            self.logger.info(f"{'='*60}")
            
            for reduction_level, keep_ratio in self.REDUCTION_LEVELS:
                if method == "fast-simplification":
                    result = self._simplify_fast_simplification(
                        input_path, keep_ratio, reduction_level
                    )
                elif method == "open3d":
                    result = self._simplify_open3d(
                        input_path, keep_ratio, reduction_level
                    )
                elif method == "meshoptimizer":
                    result = self._simplify_meshoptimizer(
                        input_path, keep_ratio, reduction_level
                    )
                elif method == "cgal":
                    result = self._simplify_cgal(
                        input_path, keep_ratio, reduction_level
                    )
                
                results.append(result)
                self.logger.info("")
        
        total_time = time.perf_counter() - total_start
        
        # Generate report
        report = BenchmarkReport(
            system_info=self._get_system_info(),
            results=results,
            total_execution_time_seconds=total_time,
            generated_at=datetime.now().isoformat()
        )
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: BenchmarkReport) -> None:
        """Print formatted summary of benchmark results."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("=" * 60)
        
        # Group results by method
        for method in ["fast-simplification", "open3d", "meshoptimizer", "cgal"]:
            method_results = [r for r in report.results if r.method == method]
            if not method_results:
                continue
            
            self.logger.info(f"\n{method.upper()}:")
            self.logger.info("-" * 55)
            self.logger.info(
                f"{'Reduction':<12} {'Faces (in→out)':<20} "
                f"{'Time (ms)':<12} {'Actual %':<10}"
            )
            self.logger.info("-" * 55)
            
            for r in method_results:
                if r.success:
                    faces_str = f"{r.input_metrics.face_count:,} → {r.output_metrics.face_count:,}"
                    self.logger.info(
                        f"{r.reduction_level:<12} {faces_str:<20} "
                        f"{r.performance.execution_time_ms:<12.2f} "
                        f"{r.actual_reduction_ratio*100:<10.1f}"
                    )
                else:
                    self.logger.info(f"{r.reduction_level:<12} FAILED: {r.error_message}")
        
        self.logger.info(f"\nTotal execution time: {report.total_execution_time_seconds:.2f} seconds")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def save_report(self, report: BenchmarkReport, output_path: Union[str, Path]) -> None:
        """Save benchmark report to JSON file."""
        output_path = Path(output_path)
        
        # Convert dataclasses to dicts
        report_dict = {
            "system_info": asdict(report.system_info),
            "results": [asdict(r) for r in report.results],
            "total_execution_time_seconds": report.total_execution_time_seconds,
            "generated_at": report.generated_at
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Report saved to: {output_path}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line entry point for mesh simplification tool."""
    parser = argparse.ArgumentParser(
        description="Mesh Simplification Comparison Tool - Compare fast-simplification "
                    "and Open3D mesh simplification libraries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fast-simplification on a mesh
  python mesh_simplifier.py --input model.ply --method fast-simplification

  # Run Open3D on a mesh
  python mesh_simplifier.py --input model.ply --method open3d

  # Run meshoptimizer on a mesh
  python mesh_simplifier.py --input model.ply --method meshoptimizer
  
  # Run CGAL on a mesh (requires compiled cgal_simplify executable)
  python mesh_simplifier.py --input model.ply --method cgal --cgal-executable ./build/bin/cgal_simplify

  # Compare all methods
  python mesh_simplifier.py --input model.ply --method all --output-dir ./results

  # Adjust parameters
  python mesh_simplifier.py --input model.ply --method all --aggressiveness 5

Supported formats: .ply, .stl, .obj, .off, .vtk
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input mesh file (.ply, .stl, .obj, .off, .vtk)"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["fast-simplification", "open3d", "meshoptimizer", "cgal", "all"],
        default="all",
        help="Simplification method to use (default: all)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./simplified_meshes",
        help="Output directory for simplified meshes (default: ./simplified_meshes)"
    )
    
    parser.add_argument(
        "--aggressiveness", "-a",
        type=int,
        default=7,
        choices=range(0, 11),
        metavar="[0-10]",
        help="fast-simplification aggressiveness parameter (0=slowest/best quality, "
             "10=fastest/lower quality, default: 7)"
    )
    
    parser.add_argument(
        "--boundary-weight", "-b",
        type=float,
        default=1.0,
        help="Open3D boundary preservation weight (default: 1.0)"
    )
    
    parser.add_argument(
        "--cgal-executable",
        type=str,
        default="./cgal_simplify",
        help="Path to CGAL simplification executable (default: ./cgal_simplify)"
    )
    
    parser.add_argument(
        "--save-report", "-r",
        type=str,
        default=None,
        help="Save JSON report to specified path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    try:
        # Create simplifier
        simplifier = MeshSimplifier(
            method=args.method,
            output_dir=args.output_dir,
            aggressiveness=args.aggressiveness,
            boundary_weight=args.boundary_weight,
            cgal_executable=args.cgal_executable,
            log_level=log_level
        )
        
        # Run benchmark
        report = simplifier.run_benchmark(args.input)
        
        # Save report if requested
        if args.save_report:
            simplifier.save_report(report, args.save_report)
        else:
            # Auto-save report to output directory
            report_path = simplifier.output_dir / "benchmark_report.json"
            simplifier.save_report(report, report_path)
        
        # Exit with success/failure based on results
        successful = sum(1 for r in report.results if r.success)
        total = len(report.results)
        
        if successful == total:
            print(f"\n✓ All {total} operations completed successfully.")
            sys.exit(0)
        else:
            print(f"\n⚠ {successful}/{total} operations completed successfully.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(4)


if __name__ == "__main__":
    main()