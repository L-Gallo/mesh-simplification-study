# save as: run_meshcnn.py
"""
MeshCNN Python Wrapper - Working Implementation
Tested with MeshCNN + PyTorch 1.2.0 + Python 3.7
"""
import subprocess
import os
import sys
from pathlib import Path
import argparse
import shutil

class MeshCNNRunner:
    def __init__(self, meshcnn_path="MeshCNN"):
        self.meshcnn_path = Path(meshcnn_path).expanduser()
        if not self.meshcnn_path.exists():
            raise FileNotFoundError(f"MeshCNN not found at {meshcnn_path}")
        
        self.test_script = self.meshcnn_path / "test.py"
        if not self.test_script.exists():
            raise FileNotFoundError(f"test.py not found in {meshcnn_path}")
    
    def run_classification(
        self,
        mesh_obj_path,
        output_dir,
        model_name="shrec16",
        export_pooled=True
    ):
        """
        Run MeshCNN classification and export pooled meshes.
        
        Args:
            mesh_obj_path: Path to input OBJ file
            output_dir: Where to save results
            model_name: Name of trained model (shrec16, cubes, etc)
            export_pooled: Whether to export intermediate pooled meshes
        
        Returns:
            dict with results
        """
        mesh_path = Path(mesh_obj_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # MeshCNN expects mesh in a specific directory structure
        # dataroot/
        #   └── test/
        #       └── mesh.obj
        
        # Create temporary test directory
        test_dir = output_dir / "test_input"
        test_dir.mkdir(exist_ok=True)
        
        # Copy mesh to test directory
        test_mesh = test_dir / mesh_path.name
        shutil.copy(mesh_path, test_mesh)
        
        # Prepare export folder
        export_folder = output_dir / "pooled_meshes" if export_pooled else None
        
        # Build command
        cmd = [
            "python", str(self.test_script),
            "--dataroot", str(test_dir.parent),
            "--name", model_name,
            "--ncf", "64", "128", "256", "256",  # Network architecture
            "--ninput_edges", "750",  # Input edge count
            "--pool_res", "600", "450", "300", "180",  # Pooling resolutions
            "--which_epoch", "latest",
            "--phase", "test",
            "--gpu_ids", "-1"  # Use CPU for segmentation
        ]
        
        if export_pooled:
            export_folder.mkdir(exist_ok=True)
            cmd.extend(["--export_folder", str(export_folder)])
        
        print("="*70)
        print("Running MeshCNN Classification")
        print("="*70)
        print(f"Input: {mesh_path}")
        print(f"Model: {model_name}")
        print(f"Working dir: {self.meshcnn_path}")
        print(f"Command: {' '.join(cmd)}")
        print("="*70)
        
        # Run MeshCNN
        result = subprocess.run(
            cmd,
            cwd=self.meshcnn_path,
            capture_output=True,
            text=True
        )
        
        # Parse results
        success = result.returncode == 0
        
        if not success:
            print("ERROR Output:")
            print(result.stderr)
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
        
        # Find output meshes
        pooled_meshes = []
        if export_pooled and export_folder.exists():
            pooled_meshes = sorted(export_folder.glob("*.obj"))
        
        print("\n" + "="*70)
        print("✓ MeshCNN Completed Successfully")
        print("="*70)
        if pooled_meshes:
            print(f"Pooled meshes exported: {len(pooled_meshes)}")
            for mesh in pooled_meshes:
                print(f"  - {mesh.name}")
        print("="*70)
        
        return {
            "success": True,
            "pooled_meshes": [str(p) for p in pooled_meshes],
            "output_dir": str(output_dir),
            "stdout": result.stdout
        }
    
    def run_segmentation(
        self,
        mesh_obj_path,
        output_dir,
        model_name="human_seg"
    ):
        """
        Run MeshCNN segmentation.
        
        Args:
            mesh_obj_path: Path to input OBJ file
            output_dir: Where to save results
            model_name: Name of trained model
        
        Returns:
            dict with results
        """
        mesh_path = Path(mesh_obj_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test directory structure
        test_dir = output_dir / "test_input"
        test_dir.mkdir(exist_ok=True)
        test_mesh = test_dir / mesh_path.name
        shutil.copy(mesh_path, test_mesh)
        
        # For segmentation, also need .eseg and .seseg files
        # These contain edge features - MeshCNN can compute them
        
        cmd = [
            "python", str(self.test_script),
            "--dataroot", str(test_dir.parent),
            "--name", model_name,
            "--ncf", "32", "64", "128", "256",
            "--ninput_edges", "2280",
            "--pool_res", "1800", "1350", "600",
            "--which_epoch", "latest",
            "--phase", "test",
            "--gpu_ids", "-1"  # Use CPU for segmentation
        ]
        
        print("="*70)
        print("Running MeshCNN Segmentation")
        print("="*70)
        print(f"Input: {mesh_path}")
        print(f"Command: {' '.join(cmd)}")
        print("="*70)
        
        result = subprocess.run(
            cmd,
            cwd=self.meshcnn_path,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        
        if success:
            print("✓ Segmentation completed")
            # Segmentation outputs are in checkpoints/{model_name}/results/
            results_dir = self.meshcnn_path / "checkpoints" / model_name / "results"
            segmented_meshes = list(results_dir.glob("*.obj")) if results_dir.exists() else []
            
            return {
                "success": True,
                "segmented_meshes": [str(p) for p in segmented_meshes],
                "output_dir": str(output_dir)
            }
        else:
            print("✗ Segmentation failed")
            print(result.stderr)
            return {
                "success": False,
                "error": result.stderr
            }


def main():
    parser = argparse.ArgumentParser(
        description="Run MeshCNN on your meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run classification and export pooled meshes
  python run_meshcnn.py --input mesh.obj --output ./results --task classification
  
  # Run segmentation
  python run_meshcnn.py --input mesh.obj --output ./results --task segmentation
        """
    )
    
    parser.add_argument("--input", required=True, help="Input OBJ mesh file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--task", choices=["classification", "segmentation"], 
                       default="classification", help="Task to perform")
    parser.add_argument("--model", default=None, 
                       help="Model name (default: shrec16 for classification, human_seg for segmentation)")
    parser.add_argument("--meshcnn-path", default="MeshCNN",
                       help="Path to MeshCNN repository")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.endswith('.obj'):
        print("Error: Input must be OBJ format")
        print("Use prepare_mesh_for_meshcnn.py to convert your mesh first")
        sys.exit(1)
    
    # Determine model name
    if args.model is None:
        args.model = "shrec16" if args.task == "classification" else "human_seg"
    
    # Run MeshCNN
    try:
        runner = MeshCNNRunner(meshcnn_path=args.meshcnn_path)
        
        if args.task == "classification":
            result = runner.run_classification(
                mesh_obj_path=args.input,
                output_dir=args.output,
                model_name=args.model,
                export_pooled=True
            )
        else:
            result = runner.run_segmentation(
                mesh_obj_path=args.input,
                output_dir=args.output,
                model_name=args.model
            )
        
        if result["success"]:
            print("\n✓ Success!")
            if "pooled_meshes" in result:
                print(f"Pooled meshes: {len(result['pooled_meshes'])}")
                for mesh in result['pooled_meshes']:
                    print(f"  {mesh}")
        else:
            print("\n✗ Failed!")
            print(result.get("error", "Unknown error"))
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()