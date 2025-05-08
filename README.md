# VGGSfM: Visual Geometry Grounded Deep Structure From Motion( 3D Reconstruction Implementation )

## Project Overview

This project implements the VGGSfM (Visual Geometry Grounded Structure from Motion) pipeline for 3D reconstruction from a set of images. VGGSfM is a state-of-the-art deep learning-based system that creates a fully differentiable and end-to-end trainable pipeline for Structure-from-Motion tasks.

The project successfully reconstructs a kitchen scene from 25 images, producing a sparse point cloud with approximately 1800 3D points and camera poses, all visualizable in standard tools like COLMAP GUI and MeshLab.

## Features

- **Deep 2D Point Tracking**: Uses advanced deep learning for pixel-accurate track generation
- **Simultaneous Camera Registration**: Recovers all camera parameters at once instead of incrementally
- **Differentiable Bundle Adjustment**: Learns optimal refinement strategies through training
- **COLMAP-Compatible Output**: Generates standard cameras.bin, images.bin and points3D.bin files
- **Visualization Support**: Compatible with MeshLab and COLMAP GUI for 3D visualization

## Environment Setup

The implementation requires the following environment:

### Hardware Requirements
- Google Colab with GPU runtime (recommended)
- For local execution: CUDA-compatible GPU recommended

### Dependencies

```bash
# Clone repositories
git clone https://github.com/facebookresearch/vggsfm.git
cd vggsfm

# System dependencies
apt-get update
apt-get install -y colmap ffmpeg

# Python dependencies
pip install -U pip
pip install opencv-python hydra-core omegaconf visdom pycolmap kornia einops
pip install git+https://github.com/cvg/LightGlue.git
pip install git+https://github.com/facebookresearch/dinov2.git

# PyTorch with CUDA
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.29

# Fix dinov2 dependency
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
sed -i 's/xformers==0.0.18/xformers>=0.0.18/' setup.py
pip install .

# Build Ceres Solver (required for triangulation)
git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build && cd build
cmake .. -DBUILD_TESTING=OFF
make -j8
make install
```

## Usage

### Running the Pipeline

```bash
# Change to the VGGSfM directory
cd path/to/vggsfm

# Run the demo on kitchen scene
python demo.py --config-path cfgs --config-name demo
```

### Key Configuration Parameters

You can adjust these parameters in the config file or via command line arguments:

```
--SCENE_DIR: Path to input images (e.g., examples/kitchen)
--query_frame_num: Number of frames to use as query (default: 1)
--max_query_pts: Maximum number of query points (default: 512)
--mixed_precision: Precision mode (fp16, None)
--robust_refine: Number of robust refinement iterations (default: 2)
--save_to_disk: Whether to save output files (default: true)
```

### For CPU-Only Environments

If running in a CPU-only environment, make these modifications:

```bash
# Force CPU loading for checkpoints
python demo.py SCENE_DIR=examples/kitchen query_frame_num=1 max_query_pts=512 mixed_precision=None
```

## Output Files

After successful execution, the following outputs are generated:

```
SCENE_DIR/
├── sparse/
│   ├── cameras.bin    # Camera intrinsics
│   ├── images.bin     # Camera extrinsics + point associations
│   └── points3D.bin   # 3D point cloud
├── sparse_ply.ply     # PLY format point cloud (for MeshLab)
```

## Visualization

### MeshLab
- Open sparse_ply.ply with MeshLab to view the 3D point cloud

### COLMAP GUI
- Open COLMAP
- File → Import Model → Path to sparse directory
- View cameras (red pyramids) and 3D points

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   - Error: "CUDA error: no kernel image is available for execution"
   - Solution: Ensure compatible PyTorch and CUDA versions

2. **Memory Errors**
   - Error: "CUDA out of memory"
   - Solution: Reduce max_query_pts parameter

3. **Missing pyceres Module**
   - Error: "ModuleNotFoundError: No module named 'pyceres'"
   - Solution: Follow the Ceres Solver build instructions carefully

4. **xformers Compatibility Issues**
   - Error: "No matching distribution found for xformers==0.0.18"
   - Solution: Patch the setup.py file as shown in the dependency setup

## References

- VGGSfM Paper: https://arxiv.org/abs/2312.04563
- Official Repository: https://github.com/facebookresearch/vggsfm
- Project Website: https://vggsfm.github.io/

## License

This implementation follows the license terms of the original VGGSfM repository.