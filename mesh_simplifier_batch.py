#!/usr/bin/env python3
"""
Batch Mesh Simplification Benchmark Tool
=========================================
Comprehensive testing framework for comparing traditional geometric mesh 
simplification methods across multiple assets, reduction levels, and repetitions.

Supports: fast-simplification, Open3D, meshoptimizer, CGAL
Features: Performance profiling, geometric accuracy, stability analysis, per-method statistics

Author: Master's Thesis Research Tool
Version: 2.0.0
"""

import argparse
import gc
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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import psutil

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

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

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError as e:
    TRIMESH_AVAILABLE = False
    TRIMESH_ERROR = str(e)


# =============================================================================
# Enums and Constants
# =============================================================================

class FailureType(Enum):
    """Types of test failures."""
    SUCCESS = "success"
    CRASH = "application_crash"
    TIMEOUT = "timeout"
    INVALID_GEOMETRY = "invalid_geometry"
    PARAMETER_FAILURE = "parameter_error"


SUPPORTED_FORMATS = {'.obj'}  # Strictly OBJ for CGAL compatibility


# =============================================================================
# Data Classes
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
    trimesh_version: Optional[str] = None


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
class GeometricAccuracyMetrics:
    """Geometric accuracy between original and simplified mesh."""
    hausdorff_distance_normalized: float  # % of bounding box diagonal
    hausdorff_distance_raw: float  # absolute units
    rmse_normalized: float  # % of bounding box diagonal
    rmse_raw: float  # absolute units
    sample_points: int
    bounding_box_diagonal: float


@dataclass
class RepetitionAnalysis:
    """Statistical analysis of repetitions for stability assessment."""
    mean_time_ms: float
    std_time_ms: float
    cv_time: float  # Coefficient of variation (%)
    time_stable: bool  # True if CV < 15%
    
    mean_memory_mb: float
    std_memory_mb: float
    cv_memory: float
    memory_stable: bool
    
    outlier_runs: List[int]  # Run numbers with Z-score > 2
    overall_stable: bool
    warning_message: Optional[str] = None


@dataclass
class SimplificationResult:
    """Complete result of a single simplification operation."""
    # Identification
    asset_name: str
    method: str
    reduction_level: str
    run_number: int  # 1, 2, or 3
    
    # Paths
    input_mesh_path: str
    output_mesh_path: str
    
    # Metrics
    target_reduction_ratio: float
    actual_reduction_ratio: float
    input_metrics: MeshMetrics
    output_metrics: MeshMetrics
    performance: PerformanceMetrics
    
    # Status (required fields)
    success: bool
    
    # Optional fields (must come after required fields)
    geometric_accuracy: Optional[GeometricAccuracyMetrics] = None
    failure_type: FailureType = FailureType.SUCCESS
    error_message: Optional[str] = None
    instability_flag: bool = False
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetResults:
    """Complete results for a single asset."""
    asset_name: str
    original_metrics: MeshMetrics
    methods: Dict[str, Dict[str, Any]]  # method -> reduction -> {repetitions, statistics}


@dataclass
class MethodStatistics:
    """Comprehensive statistics for a single method."""
    method_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    unstable_tests: int
    success_rate: float
    stability_rate: float
    
    # Failure breakdown
    crashes: int
    timeouts: int
    invalid_geometry: int
    
    # Performance aggregates
    mean_time_ms: float
    std_time_ms: float
    mean_memory_mb: float
    std_memory_mb: float
    
    # Geometric accuracy aggregates
    mean_hausdorff: float
    std_hausdorff: float
    mean_rmse: float
    std_rmse: float
    
    # Patterns
    problematic_assets: List[str]
    problematic_reductions: List[str]


@dataclass
class BatchReport:
    """Complete benchmark report for entire batch."""
    system_info: SystemInfo
    total_assets: int
    total_tests: int
    total_execution_time_seconds: float
    generated_at: str
    
    # Asset-level results
    assets: Dict[str, AssetResults]
    
    # Method-level statistics
    method_statistics: Dict[str, MethodStatistics]
    
    # Overall statistics
    overall_success_rate: float
    overall_stability_rate: float
    total_failures: int
    total_unstable: int


# =============================================================================
# Memory Profiler
# =============================================================================

class MemoryProfiler:
    """
    Memory profiler combining tracemalloc (Python allocations) and psutil 
    (total process memory including C++ libraries).
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
# Statistical Analysis Functions
# =============================================================================

def identify_outliers_zscore(values: List[float], threshold: float = 2.0) -> List[int]:
    """
    Identify outliers using Z-score method.
    Returns indices of values with |Z-score| > threshold.
    """
    if len(values) < 3:
        return []
    
    values_arr = np.array(values)
    mean = np.mean(values_arr)
    std = np.std(values_arr)
    
    if std == 0:
        return []
    
    z_scores = np.abs((values_arr - mean) / std)
    outliers = np.where(z_scores > threshold)[0].tolist()
    
    return outliers


def analyze_repetitions(results: List[SimplificationResult]) -> RepetitionAnalysis:
    """
    Analyze 3 repetitions for stability.
    Flags tests with CV > 15% or outliers with Z-score > 2.
    """
    if not results or len(results) < 2:
        return RepetitionAnalysis(
            mean_time_ms=0, std_time_ms=0, cv_time=0, time_stable=True,
            mean_memory_mb=0, std_memory_mb=0, cv_memory=0, memory_stable=True,
            outlier_runs=[], overall_stable=True
        )
    
    # Extract metrics from successful runs only
    successful = [r for r in results if r.success]
    if not successful:
        return RepetitionAnalysis(
            mean_time_ms=0, std_time_ms=0, cv_time=0, time_stable=False,
            mean_memory_mb=0, std_memory_mb=0, cv_memory=0, memory_stable=False,
            outlier_runs=[], overall_stable=False,
            warning_message="All repetitions failed"
        )
    
    times = [r.performance.execution_time_ms for r in successful]
    memories = [r.performance.peak_memory_mb for r in successful]
    
    # Calculate coefficient of variation
    mean_time = np.mean(times)
    std_time = np.std(times)
    cv_time = (std_time / mean_time * 100) if mean_time > 0 else 0
    
    mean_memory = np.mean(memories)
    std_memory = np.std(memories)
    cv_memory = (std_memory / mean_memory * 100) if mean_memory > 0 else 0
    
    # Identify outliers
    time_outliers = identify_outliers_zscore(times)
    memory_outliers = identify_outliers_zscore(memories)
    outlier_runs = sorted(set(time_outliers + memory_outliers))
    
    # Stability thresholds
    time_stable = cv_time < 15.0
    memory_stable = cv_memory < 15.0
    overall_stable = time_stable and memory_stable and not outlier_runs
    
    # Generate warning message
    warning = None
    if not overall_stable:
        warnings = []
        if not time_stable:
            warnings.append(f"High time variance (CV={cv_time:.1f}%)")
        if not memory_stable:
            warnings.append(f"High memory variance (CV={cv_memory:.1f}%)")
        if outlier_runs:
            warnings.append(f"Outlier runs: {[r+1 for r in outlier_runs]}")
        warning = "; ".join(warnings)
    
    return RepetitionAnalysis(
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        cv_time=cv_time,
        time_stable=time_stable,
        mean_memory_mb=mean_memory,
        std_memory_mb=std_memory,
        cv_memory=cv_memory,
        memory_stable=memory_stable,
        outlier_runs=outlier_runs,
        overall_stable=overall_stable,
        warning_message=warning
    )


# =============================================================================
# Batch Mesh Simplifier
# =============================================================================

class BatchMeshSimplifier:
    """
    Batch processor for mesh simplification across multiple assets and methods.
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        methods: List[str],
        reduction_levels: List[int],
        repetitions: int = 3,
        compute_accuracy: bool = True,
        aggressiveness: int = 7,
        boundary_weight: float = 1.0,
        cgal_executable: Path = Path("./cgal_simplify"),
        log_level: int = logging.INFO
    ):
        """
        Initialize batch simplifier.
        
        Args:
            input_dir: Directory containing input OBJ files
            output_dir: Directory for output files
            methods: List of methods to test
            reduction_levels: Reduction percentages (e.g., [75, 50, 25])
            repetitions: Number of repetitions per test (default: 3)
            compute_accuracy: Whether to compute geometric accuracy
            aggressiveness: fast-simplification parameter
            boundary_weight: Open3D parameter
            cgal_executable: Path to CGAL executable
            log_level: Logging level
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.methods = methods
        self.reduction_levels = reduction_levels
        self.repetitions = repetitions
        self.compute_accuracy = compute_accuracy
        self.aggressiveness = aggressiveness
        self.boundary_weight = boundary_weight
        self.cgal_executable = cgal_executable
        
        # Create output directory FIRST (before logging setup)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup dual logging (console + file)
        self.logger = self._setup_logging(log_level)
        
        # Validate libraries
        self._validate_libraries()
        
        # Storage for all results
        self.all_results: List[SimplificationResult] = []
    
    def _setup_logging(self, log_level: int) -> logging.Logger:
        """Setup dual logging to console and file."""
        logger = logging.getLogger("BatchSimplifier")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        # Console handler (INFO level, clean format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (DEBUG level, detailed format)
        log_file = self.output_dir / 'benchmark_log.txt'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _validate_libraries(self) -> None:
        """Validate that required libraries are available."""
        if "fast-simplification" in self.methods:
            if not FAST_SIMPLIFICATION_AVAILABLE:
                raise ImportError(
                    f"fast-simplification not available: {FAST_SIMPLIFICATION_ERROR}\n"
                    "Install with: pip install pyvista fast-simplification"
                )
        
        if "open3d" in self.methods:
            if not OPEN3D_AVAILABLE:
                raise ImportError(
                    f"Open3D not available: {OPEN3D_ERROR}\n"
                    "Install with: pip install open3d"
                )
        
        if "meshoptimizer" in self.methods:
            if not MESHOPTIMIZER_AVAILABLE:
                raise ImportError(
                    f"meshoptimizer not available: {MESHOPTIMIZER_ERROR}\n"
                    "Install with: pip install meshoptimizer"
                )
        
        if "cgal" in self.methods:
            if not self.cgal_executable.exists():
                raise FileNotFoundError(
                    f"CGAL executable not found: {self.cgal_executable}\n"
                    "Please compile cgal_simplify and provide path via --cgal-executable"
                )
        
        if self.compute_accuracy and not TRIMESH_AVAILABLE:
            raise ImportError(
                f"trimesh not available: {TRIMESH_ERROR}\n"
                "Install with: pip install trimesh"
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
        
        if TRIMESH_AVAILABLE:
            info.trimesh_version = trimesh.__version__
        
        return info
    
    def _get_mesh_metrics_pyvista(
        self, 
        mesh: "pv.PolyData", 
        filepath: Optional[Path] = None
    ) -> MeshMetrics:
        """Extract metrics from PyVista mesh."""
        file_size = filepath.stat().st_size if filepath and filepath.exists() else None
        
        bounds = mesh.bounds
        diagonal = np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
        
        face_count = mesh.n_cells if hasattr(mesh, 'n_cells') else mesh.n_faces
        
        return MeshMetrics(
            vertex_count=mesh.n_points,
            face_count=face_count,
            file_size_bytes=file_size,
            bounding_box_diagonal=float(diagonal)
        )
    
    def _get_mesh_metrics_open3d(
        self, 
        mesh: "o3d.geometry.TriangleMesh", 
        filepath: Optional[Path] = None
    ) -> MeshMetrics:
        """Extract metrics from Open3D mesh."""
        file_size = filepath.stat().st_size if filepath and filepath.exists() else None
        
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        diagonal = np.sqrt(extent[0]**2 + extent[1]**2 + extent[2]**2)
        
        return MeshMetrics(
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.triangles),
            file_size_bytes=file_size,
            bounding_box_diagonal=float(diagonal)
        )
    
    def _compute_geometric_accuracy(
        self,
        original_path: Path,
        simplified_path: Path,
        n_samples: int = 10000
    ) -> Optional[GeometricAccuracyMetrics]:
        """
        Compute Hausdorff distance and RMSE using trimesh.
        Per methodology Section 3.3.
        """
        if not TRIMESH_AVAILABLE:
            return None
        
        try:
            # Load meshes
            mesh_orig = trimesh.load(str(original_path), force='mesh')
            mesh_simp = trimesh.load(str(simplified_path), force='mesh')
            
            # Handle Scene objects
            if isinstance(mesh_orig, trimesh.Scene):
                mesh_orig = list(mesh_orig.geometry.values())[0]
            if isinstance(mesh_simp, trimesh.Scene):
                mesh_simp = list(mesh_simp.geometry.values())[0]
            
            # Bounding box diagonal for normalization
            bb_diagonal = np.linalg.norm(mesh_orig.bounds[1] - mesh_orig.bounds[0])
            
            # Sample points uniformly from surfaces
            points_orig, _ = trimesh.sample.sample_surface(mesh_orig, n_samples)
            points_simp, _ = trimesh.sample.sample_surface(mesh_simp, n_samples)
            
            # Two-way Hausdorff distance
            # Direction 1: orig -> simp
            query_1 = trimesh.proximity.ProximityQuery(mesh_simp)
            distances_1 = np.abs(query_1.signed_distance(points_orig))
            hausdorff_1 = distances_1.max()
            
            # Direction 2: simp -> orig
            query_2 = trimesh.proximity.ProximityQuery(mesh_orig)
            distances_2 = np.abs(query_2.signed_distance(points_simp))
            hausdorff_2 = distances_2.max()
            
            hausdorff_raw = max(hausdorff_1, hausdorff_2)
            hausdorff_norm = (hausdorff_raw / bb_diagonal) * 100
            
            # RMSE: simplified -> original
            rmse_raw = np.sqrt(np.mean(distances_2 ** 2))
            rmse_norm = (rmse_raw / bb_diagonal) * 100
            
            return GeometricAccuracyMetrics(
                hausdorff_distance_normalized=float(hausdorff_norm),
                hausdorff_distance_raw=float(hausdorff_raw),
                rmse_normalized=float(rmse_norm),
                rmse_raw=float(rmse_raw),
                sample_points=n_samples,
                bounding_box_diagonal=float(bb_diagonal)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to compute geometric accuracy: {e}")
            return None
    
    def _simplify_fast_simplification(
        self,
        input_path: Path,
        output_path: Path,
        target_keep_ratio: float,
        reduction_level: str,
        run_number: int,
        compute_accuracy: bool
    ) -> SimplificationResult:
        """Simplify using fast-simplification."""
        asset_name = input_path.stem
        profiler = MemoryProfiler()
        
        try:
            # Load mesh
            mesh = pv.read(str(input_path))
            input_metrics = self._get_mesh_metrics_pyvista(mesh, input_path)
            
            # Calculate target_reduction for fast-simplification
            target_reduction = 1.0 - target_keep_ratio
            
            # Profile simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
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
            simplified.save(str(output_path))
            output_metrics = self._get_mesh_metrics_pyvista(simplified, output_path)
            
            # Calculate actual reduction
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            # Compute geometric accuracy (only for run1)
            geometric_accuracy = None
            if compute_accuracy and run_number == 1:
                geometric_accuracy = self._compute_geometric_accuracy(
                    input_path, output_path
                )
            
            # Clear memory
            del mesh, simplified
            gc.collect()
            
            return SimplificationResult(
                asset_name=asset_name,
                method="fast-simplification",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
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
                geometric_accuracy=geometric_accuracy,
                success=True,
                failure_type=FailureType.SUCCESS,
                additional_params={"aggressiveness": self.aggressiveness}
            )
            
        except Exception as e:
            return SimplificationResult(
                asset_name=asset_name,
                method="fast-simplification",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                failure_type=FailureType.CRASH,
                error_message=str(e)
            )
    
    def _simplify_open3d(
        self,
        input_path: Path,
        output_path: Path,
        target_keep_ratio: float,
        reduction_level: str,
        run_number: int,
        compute_accuracy: bool
    ) -> SimplificationResult:
        """Simplify using Open3D."""
        asset_name = input_path.stem
        profiler = MemoryProfiler()
        
        try:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            
            # Preprocessing
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            
            input_metrics = self._get_mesh_metrics_open3d(mesh, input_path)
            
            # Calculate target triangle count
            target_triangles = int(input_metrics.face_count * target_keep_ratio)
            target_triangles = max(target_triangles, 4)
            
            # Profile simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
            simplified = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles,
                maximum_error=float('inf'),
                boundary_weight=self.boundary_weight
            )
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            execution_time = end_time - start_time
            
            # Compute normals
            simplified.compute_vertex_normals()
            
            # Save
            o3d.io.write_triangle_mesh(str(output_path), simplified)
            output_metrics = self._get_mesh_metrics_open3d(simplified, output_path)
            
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            # Geometric accuracy
            geometric_accuracy = None
            if compute_accuracy and run_number == 1:
                geometric_accuracy = self._compute_geometric_accuracy(
                    input_path, output_path
                )
            
            del mesh, simplified
            gc.collect()
            
            return SimplificationResult(
                asset_name=asset_name,
                method="open3d",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
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
                geometric_accuracy=geometric_accuracy,
                success=True,
                failure_type=FailureType.SUCCESS,
                additional_params={
                    "boundary_weight": self.boundary_weight,
                    "target_triangles": target_triangles
                }
            )
            
        except Exception as e:
            return SimplificationResult(
                asset_name=asset_name,
                method="open3d",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                failure_type=FailureType.CRASH,
                error_message=str(e)
            )
    
    def _simplify_meshoptimizer(
        self,
        input_path: Path,
        output_path: Path,
        target_keep_ratio: float,
        reduction_level: str,
        run_number: int,
        compute_accuracy: bool
    ) -> SimplificationResult:
        """Simplify using meshoptimizer."""
        asset_name = input_path.stem
        profiler = MemoryProfiler()
        
        try:
            # Load with PyVista
            self.logger.debug(f"[meshoptimizer] Loading mesh: {input_path}")
            mesh = pv.read(str(input_path))
            input_metrics = self._get_mesh_metrics_pyvista(mesh, input_path)
            
            # Extract vertex and face data
            self.logger.debug(f"[meshoptimizer] Extracting vertices and faces")
            vertices = np.array(mesh.points, dtype=np.float32)
            self.logger.debug(f"[meshoptimizer] Vertices shape: {vertices.shape}, dtype: {vertices.dtype}")
            
            faces = mesh.faces.reshape(-1, 4)[:, 1:4].astype(np.uint32)
            self.logger.debug(f"[meshoptimizer] Faces shape: {faces.shape}, dtype: {faces.dtype}")
            
            # Calculate target face count
            target_faces = int(len(faces) * target_keep_ratio)
            target_faces = max(target_faces, 4)
            target_indices = target_faces * 3  # 3 indices per triangle
            
            self.logger.debug(f"[meshoptimizer] Target faces: {target_faces}, target_indices: {target_indices}")
            self.logger.debug(f"[meshoptimizer] Input faces.flatten() shape: {faces.flatten().shape}")
            
            # Profile simplification
            memory_before = profiler.start()
            start_time = time.perf_counter()
            
            # meshoptimizer API: simplify(destination, indices, vertex_positions, ...)
            # Need to pre-allocate destination array
            self.logger.debug(f"[meshoptimizer] Calling meshoptimizer.simplify()...")
            self.logger.debug(f"[meshoptimizer]   indices type: {type(faces.flatten())}, shape: {faces.flatten().shape}")
            self.logger.debug(f"[meshoptimizer]   vertices type: {type(vertices)}, shape: {vertices.shape}")
            self.logger.debug(f"[meshoptimizer]   target_indices: {target_indices}")
            
            # Pre-allocate destination array (max possible size = input size)
            indices_flat = faces.flatten()
            destination = np.zeros(len(indices_flat), dtype=np.uint32)
            
            # Call simplify with correct parameter order
            result_count = meshoptimizer.simplify(
                destination,           # destination: pre-allocated output array
                indices_flat,          # indices: input triangle indices
                vertices,              # vertex_positions: (N, 3) array
                target_index_count=target_indices,  # target_index_count: desired output size
                target_error=1e-2      # target_error: error threshold
            )
            
            self.logger.debug(f"[meshoptimizer] Simplify returned {result_count} indices")
            
            # Extract only the valid indices (first result_count elements)
            simplified_indices = destination[:result_count]
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            execution_time = end_time - start_time
            
            self.logger.debug(f"[meshoptimizer] Simplified indices length: {len(simplified_indices)}")
            
            simplified_faces = simplified_indices.reshape(-1, 3)
            self.logger.debug(f"[meshoptimizer] Simplified faces shape: {simplified_faces.shape}")
            
            # Remap vertices (use original 2D vertices array)
            used_vertices = np.unique(simplified_faces)
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            new_vertices = vertices[used_vertices]  # vertices is still 2D (N, 3)
            new_faces = np.array([[vertex_map[idx] for idx in face] for face in simplified_faces])
            
            self.logger.debug(f"[meshoptimizer] Remapped vertices: {len(new_vertices)}, faces: {len(new_faces)}")
            
            # Create PyVista mesh and save
            faces_with_count = np.column_stack([
                np.full(len(new_faces), 3),
                new_faces
            ]).flatten()
            simplified_mesh = pv.PolyData(new_vertices, faces_with_count)
            simplified_mesh.save(str(output_path))
            
            self.logger.debug(f"[meshoptimizer] Saved to: {output_path}")
            
            output_metrics = self._get_mesh_metrics_pyvista(simplified_mesh, output_path)
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            # Geometric accuracy
            geometric_accuracy = None
            if compute_accuracy and run_number == 1:
                geometric_accuracy = self._compute_geometric_accuracy(
                    input_path, output_path
                )
            
            del mesh, simplified_mesh
            gc.collect()
            
            return SimplificationResult(
                asset_name=asset_name,
                method="meshoptimizer",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
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
                geometric_accuracy=geometric_accuracy,
                success=True,
                failure_type=FailureType.SUCCESS,
                additional_params={"target_indices": target_indices}
            )
            
        except Exception as e:
            # Detailed error logging
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f"[meshoptimizer] DETAILED ERROR:")
            self.logger.error(f"[meshoptimizer] Error message: {str(e)}")
            self.logger.error(f"[meshoptimizer] Error type: {type(e).__name__}")
            self.logger.error(f"[meshoptimizer] Full traceback:\n{error_trace}")
            
            return SimplificationResult(
                asset_name=asset_name,
                method="meshoptimizer",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                failure_type=FailureType.CRASH,
                error_message=str(e)
            )
    
    def _simplify_cgal(
        self,
        input_path: Path,
        output_path: Path,
        target_keep_ratio: float,
        reduction_level: str,
        run_number: int,
        compute_accuracy: bool
    ) -> SimplificationResult:
        """Simplify using CGAL."""
        asset_name = input_path.stem
        profiler = MemoryProfiler()
        
        # Temp stats file
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        
        try:
            # Get input metrics
            mesh = pv.read(str(input_path))
            input_metrics = self._get_mesh_metrics_pyvista(mesh, input_path)
            del mesh
            
            # Profile simplification
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
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.perf_counter()
            memory_after, memory_delta, peak_memory = profiler.stop()
            execution_time = end_time - start_time
            
            if result.returncode != 0:
                error_msg = f"CGAL failed (code {result.returncode}): {result.stderr}"
                raise RuntimeError(error_msg)
            
            # Get output metrics
            if not output_path.exists():
                raise FileNotFoundError(f"CGAL did not create output: {output_path}")
            
            simplified = pv.read(str(output_path))
            output_metrics = self._get_mesh_metrics_pyvista(simplified, output_path)
            del simplified
            
            actual_reduction = 1.0 - (output_metrics.face_count / input_metrics.face_count)
            
            # Geometric accuracy
            geometric_accuracy = None
            if compute_accuracy and run_number == 1:
                geometric_accuracy = self._compute_geometric_accuracy(
                    input_path, output_path
                )
            
            # Clean up temp stats file
            if stats_path.exists():
                stats_path.unlink()
            
            gc.collect()
            
            return SimplificationResult(
                asset_name=asset_name,
                method="cgal",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=actual_reduction,
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
                geometric_accuracy=geometric_accuracy,
                success=True,
                failure_type=FailureType.SUCCESS
            )
            
        except subprocess.TimeoutExpired:
            return SimplificationResult(
                asset_name=asset_name,
                method="cgal",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                failure_type=FailureType.TIMEOUT,
                error_message="Process timeout (>10 minutes)"
            )
        except Exception as e:
            return SimplificationResult(
                asset_name=asset_name,
                method="cgal",
                reduction_level=reduction_level,
                run_number=run_number,
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_reduction_ratio=1.0 - target_keep_ratio,
                actual_reduction_ratio=0.0,
                input_metrics=MeshMetrics(0, 0),
                output_metrics=MeshMetrics(0, 0),
                performance=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                success=False,
                failure_type=FailureType.CRASH,
                error_message=str(e)
            )
    
    def _process_single_test(
        self,
        input_path: Path,
        method: str,
        reduction_pct: int,
        run_number: int
    ) -> SimplificationResult:
        """Process a single test (one method, one reduction, one repetition)."""
        # Convert reduction percentage to keep ratio
        keep_ratio = (100 - reduction_pct) / 100
        reduction_level = f"{reduction_pct}%"
        
        # Generate output path
        asset_name = input_path.stem
        asset_output_dir = self.output_dir / asset_name / method
        asset_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{asset_name}_{method[:5]}_{reduction_pct}pct_run{run_number}.obj"
        output_path = asset_output_dir / output_filename
        
        # Determine if we should compute accuracy (only run1)
        compute_accuracy = self.compute_accuracy and run_number == 1
        
        # Call appropriate method
        if method == "fast-simplification":
            return self._simplify_fast_simplification(
                input_path, output_path, keep_ratio, reduction_level,
                run_number, compute_accuracy
            )
        elif method == "open3d":
            return self._simplify_open3d(
                input_path, output_path, keep_ratio, reduction_level,
                run_number, compute_accuracy
            )
        elif method == "meshoptimizer":
            return self._simplify_meshoptimizer(
                input_path, output_path, keep_ratio, reduction_level,
                run_number, compute_accuracy
            )
        elif method == "cgal":
            return self._simplify_cgal(
                input_path, output_path, keep_ratio, reduction_level,
                run_number, compute_accuracy
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run_batch(self) -> BatchReport:
        """Run complete batch benchmark."""
        # Find all OBJ files
        mesh_files = sorted([
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_FORMATS
        ])
        
        if not mesh_files:
            raise FileNotFoundError(
                f"No .obj files found in {self.input_dir}"
            )
        
        # Log header
        self.logger.info("=" * 80)
        self.logger.info("BATCH MESH SIMPLIFICATION BENCHMARK")
        self.logger.info("=" * 80)
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Assets: {len(mesh_files)}")
        self.logger.info(f"Methods: {', '.join(self.methods)}")
        self.logger.info(f"Reduction levels: {', '.join(map(str, self.reduction_levels))}%")
        self.logger.info(f"Repetitions: {self.repetitions}")
        self.logger.info(f"Geometric accuracy: {'Enabled (Run1 only)' if self.compute_accuracy else 'Disabled'}")
        self.logger.info("")
        
        # Calculate total operations
        total_ops = len(mesh_files) * len(self.methods) * len(self.reduction_levels) * self.repetitions
        self.logger.info(f"Total operations: {total_ops}")
        self.logger.info("")
        
        # Progress tracking
        batch_start_time = time.perf_counter()
        
        # Setup progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_ops, desc="Benchmark", unit="test")
        else:
            pbar = None
        
        # Process each asset
        for asset_idx, mesh_file in enumerate(mesh_files, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"ASSET {asset_idx}/{len(mesh_files)}: {mesh_file.name}")
            self.logger.info(f"{'='*80}")
            
            # Get original mesh metrics
            try:
                original_mesh = pv.read(str(mesh_file))
                original_metrics = self._get_mesh_metrics_pyvista(original_mesh, mesh_file)
                self.logger.info(
                    f"Original: {original_metrics.vertex_count:,} vertices, "
                    f"{original_metrics.face_count:,} faces"
                )
                del original_mesh
            except Exception as e:
                self.logger.error(f"Failed to load {mesh_file.name}: {e}")
                continue
            
            # Process all method/reduction/repetition combinations
            for method in self.methods:
                self.logger.info(f"\n  [{method.upper()}]")
                
                for reduction_pct in self.reduction_levels:
                    self.logger.info(f"    {reduction_pct}% reduction:")
                    
                    # Run all repetitions
                    repetition_results = []
                    for rep in range(1, self.repetitions + 1):
                        result = self._process_single_test(
                            mesh_file, method, reduction_pct, rep
                        )
                        repetition_results.append(result)
                        self.all_results.append(result)
                        
                        # Log result
                        if result.success:
                            accuracy_str = ""
                            if result.geometric_accuracy:
                                accuracy_str = (
                                    f" | Hausdorff: {result.geometric_accuracy.hausdorff_distance_normalized:.3f}% "
                                    f"| RMSE: {result.geometric_accuracy.rmse_normalized:.3f}%"
                                )
                            
                            self.logger.info(
                                f"      Run {rep}: {result.performance.execution_time_ms:.1f}ms "
                                f"| {result.performance.peak_memory_mb:.1f}MB"
                                f"{accuracy_str} [OK]"
                            )
                        else:
                            self.logger.error(
                                f"      Run {rep}: FAILED - {result.error_message}"
                            )
                        
                        # Update progress bar
                        if pbar:
                            success_rate = (
                                sum(1 for r in self.all_results if r.success) / 
                                len(self.all_results) * 100
                            )
                            pbar.update(1)
                            pbar.set_postfix({
                                'asset': mesh_file.stem[:10],
                                'method': method[:5],
                                'success': f"{success_rate:.1f}%"
                            })
                    
                    # Analyze repetitions
                    analysis = analyze_repetitions(repetition_results)
                    
                    # Log statistics
                    if analysis.overall_stable:
                        self.logger.info(
                            f"      Stats: {analysis.mean_time_ms:.1f}ms +/- {analysis.std_time_ms:.1f}ms "
                            f"(CV={analysis.cv_time:.1f}%) [STABLE]"
                        )
                    else:
                        self.logger.warning(
                            f"      Stats: {analysis.mean_time_ms:.1f}ms +/- {analysis.std_time_ms:.1f}ms "
                            f"(CV={analysis.cv_time:.1f}%) [WARNING] {analysis.warning_message}"
                        )
                        
                        # Flag all repetitions as unstable
                        for result in repetition_results:
                            result.instability_flag = True
        
        if pbar:
            pbar.close()
        
        batch_end_time = time.perf_counter()
        total_time = batch_end_time - batch_start_time
        
        # Generate comprehensive report
        report = self._generate_report(total_time)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _calculate_method_statistics(self) -> Dict[str, MethodStatistics]:
        """Calculate per-method statistics."""
        stats = {}
        
        for method in self.methods:
            method_results = [r for r in self.all_results if r.method == method]
            
            if not method_results:
                continue
            
            successes = [r for r in method_results if r.success]
            failures = [r for r in method_results if not r.success]
            unstable = [r for r in method_results if r.instability_flag]
            
            # Performance aggregates (successful tests only)
            times = [r.performance.execution_time_ms for r in successes]
            memories = [r.performance.peak_memory_mb for r in successes]
            
            # Geometric accuracy (run1 successful tests only)
            run1_with_accuracy = [
                r for r in successes 
                if r.run_number == 1 and r.geometric_accuracy is not None
            ]
            
            hausdorffs = [r.geometric_accuracy.hausdorff_distance_normalized 
                         for r in run1_with_accuracy]
            rmses = [r.geometric_accuracy.rmse_normalized 
                    for r in run1_with_accuracy]
            
            # Identify problematic patterns
            problematic_assets = list(set([
                r.asset_name for r in failures
            ]))
            
            problematic_reductions = list(set([
                r.reduction_level for r in failures
            ]))
            
            stats[method] = MethodStatistics(
                method_name=method,
                total_tests=len(method_results),
                successful_tests=len(successes),
                failed_tests=len(failures),
                unstable_tests=len(unstable),
                success_rate=(len(successes) / len(method_results) * 100) if method_results else 0,
                stability_rate=((len(method_results) - len(unstable)) / len(method_results) * 100) if method_results else 0,
                crashes=sum(1 for f in failures if f.failure_type == FailureType.CRASH),
                timeouts=sum(1 for f in failures if f.failure_type == FailureType.TIMEOUT),
                invalid_geometry=sum(1 for f in failures if f.failure_type == FailureType.INVALID_GEOMETRY),
                mean_time_ms=float(np.mean(times)) if times else 0,
                std_time_ms=float(np.std(times)) if times else 0,
                mean_memory_mb=float(np.mean(memories)) if memories else 0,
                std_memory_mb=float(np.std(memories)) if memories else 0,
                mean_hausdorff=float(np.mean(hausdorffs)) if hausdorffs else 0,
                std_hausdorff=float(np.std(hausdorffs)) if hausdorffs else 0,
                mean_rmse=float(np.mean(rmses)) if rmses else 0,
                std_rmse=float(np.std(rmses)) if rmses else 0,
                problematic_assets=problematic_assets,
                problematic_reductions=problematic_reductions
            )
        
        return stats
    
    def _generate_report(self, total_time: float) -> BatchReport:
        """Generate comprehensive batch report."""
        # Calculate method statistics
        method_stats = self._calculate_method_statistics()
        
        # Overall statistics
        total_tests = len(self.all_results)
        successful = sum(1 for r in self.all_results if r.success)
        unstable = sum(1 for r in self.all_results if r.instability_flag)
        
        overall_success_rate = (successful / total_tests * 100) if total_tests > 0 else 0
        overall_stability_rate = ((total_tests - unstable) / total_tests * 100) if total_tests > 0 else 0
        
        # Group results by asset
        assets_dict = {}
        asset_names = sorted(set([r.asset_name for r in self.all_results]))
        
        for asset_name in asset_names:
            asset_results = [r for r in self.all_results if r.asset_name == asset_name]
            
            # Get original metrics (from first result)
            original_metrics = asset_results[0].input_metrics if asset_results else MeshMetrics(0, 0)
            
            # Group by method and reduction
            methods_dict = {}
            for method in self.methods:
                method_results = [r for r in asset_results if r.method == method]
                
                reductions_dict = {}
                for reduction_pct in self.reduction_levels:
                    reduction_str = f"{reduction_pct}%"
                    reduction_results = [
                        r for r in method_results 
                        if r.reduction_level == reduction_str
                    ]
                    
                    if reduction_results:
                        # Analyze repetitions
                        analysis = analyze_repetitions(reduction_results)
                        
                        reductions_dict[reduction_str] = {
                            "repetitions": [asdict(r) for r in reduction_results],
                            "statistics": asdict(analysis)
                        }
                
                if reductions_dict:
                    methods_dict[method] = reductions_dict
            
            assets_dict[asset_name] = AssetResults(
                asset_name=asset_name,
                original_metrics=original_metrics,
                methods=methods_dict
            )
        
        return BatchReport(
            system_info=self._get_system_info(),
            total_assets=len(asset_names),
            total_tests=total_tests,
            total_execution_time_seconds=total_time,
            generated_at=datetime.now().isoformat(),
            assets=assets_dict,
            method_statistics=method_stats,
            overall_success_rate=overall_success_rate,
            overall_stability_rate=overall_stability_rate,
            total_failures=total_tests - successful,
            total_unstable=unstable
        )
    
    def _print_summary(self, report: BatchReport) -> None:
        """Print formatted summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("=" * 80)
        
        # Overall statistics
        self.logger.info(f"\nTotal Tests: {report.total_tests}")
        self.logger.info(f"Execution Time: {report.total_execution_time_seconds:.1f}s ({report.total_execution_time_seconds/60:.1f} minutes)")
        self.logger.info(f"\nOverall Success Rate: {report.overall_success_rate:.1f}% ({report.total_tests - report.total_failures}/{report.total_tests})")
        self.logger.info(f"Overall Stability Rate: {report.overall_stability_rate:.1f}% ({report.total_tests - report.total_unstable}/{report.total_tests})")
        
        # Per-method success rates
        self.logger.info("\n" + "-" * 80)
        self.logger.info("METHOD COMPARISON")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Method':<25} {'Success Rate':<15} {'Stability':<15} {'Mean Time':<15}")
        self.logger.info("-" * 80)
        
        for method, stats in report.method_statistics.items():
            self.logger.info(
                f"{method:<25} "
                f"{stats.success_rate:>6.1f}% ({stats.successful_tests}/{stats.total_tests}){'':<3} "
                f"{stats.stability_rate:>6.1f}%{'':<8} "
                f"{stats.mean_time_ms:>8.1f}ms"
            )
        
        # Failure details
        if report.total_failures > 0:
            self.logger.info("\n" + "-" * 80)
            self.logger.info("FAILURE DETAILS")
            self.logger.info("-" * 80)
            
            for method, stats in report.method_statistics.items():
                if stats.failed_tests > 0:
                    failure_details = []
                    if stats.crashes > 0:
                        failure_details.append(f"{stats.crashes} crashes")
                    if stats.timeouts > 0:
                        failure_details.append(f"{stats.timeouts} timeouts")
                    if stats.invalid_geometry > 0:
                        failure_details.append(f"{stats.invalid_geometry} invalid geometry")
                    
                    self.logger.info(f"{method}: {', '.join(failure_details)}")
                    
                    if stats.problematic_assets:
                        self.logger.info(f"  Problematic assets: {', '.join(stats.problematic_assets)}")
        
        # Geometric accuracy comparison
        if self.compute_accuracy:
            self.logger.info("\n" + "-" * 80)
            self.logger.info("GEOMETRIC ACCURACY (Mean +/- SD, normalized %)")
            self.logger.info("-" * 80)
            self.logger.info(f"{'Method':<25} {'Hausdorff Distance':<30} {'RMSE':<30}")
            self.logger.info("-" * 80)
            
            for method, stats in report.method_statistics.items():
                if stats.mean_hausdorff > 0:
                    self.logger.info(
                        f"{method:<25} "
                        f"{stats.mean_hausdorff:>6.3f}% +/- {stats.std_hausdorff:>6.3f}%{'':<14} "
                        f"{stats.mean_rmse:>6.3f}% +/- {stats.std_rmse:>6.3f}%"
                    )
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        self.logger.info("=" * 80)
    
    def save_report(self, report: BatchReport, filename: str = "batch_report.json") -> None:
        """Save batch report to JSON."""
        output_path = self.output_dir / filename
        
        # Convert to dict (handle nested dataclasses)
        report_dict = {
            "system_info": asdict(report.system_info),
            "summary": {
                "total_assets": report.total_assets,
                "total_tests": report.total_tests,
                "total_execution_time_seconds": report.total_execution_time_seconds,
                "overall_success_rate": report.overall_success_rate,
                "overall_stability_rate": report.overall_stability_rate,
                "total_failures": report.total_failures,
                "total_unstable": report.total_unstable,
                "generated_at": report.generated_at
            },
            "method_statistics": {
                method: asdict(stats) 
                for method, stats in report.method_statistics.items()
            },
            "assets": {
                name: {
                    "asset_name": asset.asset_name,
                    "original_metrics": asdict(asset.original_metrics),
                    "methods": asset.methods
                }
                for name, asset in report.assets.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"\nReport saved to: {output_path}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Batch Mesh Simplification Benchmark - Compare methods across multiple assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on all meshes in directory
  python mesh_simplifier_batch.py -i ./meshes -o ./results --methods all

  # Test specific methods with custom reduction levels
  python mesh_simplifier_batch.py -i ./meshes -o ./results \\
      --methods fast-simplification open3d --reduction-levels 90 75 50 25

  # Full benchmark with geometric accuracy
  python mesh_simplifier_batch.py -i ./meshes -o ./results \\
      --methods all --reduction-levels 75 50 25 --repetitions 3 --compute-accuracy

  # Quick performance test (no accuracy, 1 repetition)
  python mesh_simplifier_batch.py -i ./meshes -o ./results \\
      --methods fast-simplification --repetitions 1 --no-compute-accuracy
        """
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing input .obj files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./benchmark_results",
        help="Output directory (default: ./benchmark_results)"
    )
    
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        choices=["fast-simplification", "open3d", "meshoptimizer", "cgal", "all"],
        default=["all"],
        help="Methods to test (default: all)"
    )
    
    parser.add_argument(
        "--reduction-levels",
        nargs="+",
        type=int,
        default=[75, 50, 25],
        help="Reduction percentages (default: 75 50 25)"
    )
    
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per test for stability analysis (default: 3)"
    )
    
    parser.add_argument(
        "--compute-accuracy",
        action="store_true",
        default=True,
        help="Compute geometric accuracy metrics (default: enabled)"
    )
    
    parser.add_argument(
        "--no-compute-accuracy",
        action="store_false",
        dest="compute_accuracy",
        help="Disable geometric accuracy computation"
    )
    
    parser.add_argument(
        "--aggressiveness", "-a",
        type=int,
        default=7,
        choices=range(0, 11),
        help="fast-simplification aggressiveness (0-10, default: 7)"
    )
    
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="Open3D boundary preservation weight (default: 1.0)"
    )
    
    parser.add_argument(
        "--cgal-executable",
        type=str,
        default="./cgal_simplify",
        help="Path to CGAL executable (default: ./cgal_simplify)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    args = parser.parse_args()
    
    # Expand "all" to actual methods
    if "all" in args.methods:
        methods = ["fast-simplification", "open3d", "meshoptimizer", "cgal"]
    else:
        methods = args.methods
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    try:
        # Create simplifier
        simplifier = BatchMeshSimplifier(
            input_dir=input_dir,
            output_dir=output_dir,
            methods=methods,
            reduction_levels=args.reduction_levels,
            repetitions=args.repetitions,
            compute_accuracy=args.compute_accuracy,
            aggressiveness=args.aggressiveness,
            boundary_weight=args.boundary_weight,
            cgal_executable=Path(args.cgal_executable),
            log_level=log_level
        )
        
        # Run batch benchmark
        report = simplifier.run_batch()
        
        # Save report
        simplifier.save_report(report)
        
        # Exit with appropriate code
        if report.total_failures == 0:
            print(f"\n[SUCCESS] All {report.total_tests} tests completed successfully.")
            sys.exit(0)
        else:
            print(f"\n[WARNING] {report.total_tests - report.total_failures}/{report.total_tests} tests successful.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()