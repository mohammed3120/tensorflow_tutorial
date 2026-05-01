# TensorFlow GPU/CPU Device Management Demo

This Jupyter notebook demonstrates how to manage and test GPU/CPU device placement in TensorFlow 2.x, including device assignment, verification, and basic tensor operations.

## 📋 System Requirements

- **OS**: Ubuntu 22.04 LTS (or compatible Linux distribution)
- **Python**: 3.10.x (3.10.12 specifically tested)
- **TensorFlow**: 2.21.0
- **CUDA**: Required for GPU support (optional, falls back to CPU)
- **Jupyter**: For running the notebook

## 🚀 Quick Setup Guide

### 1. Clone/Download the Repository
```bash
git clone https://github.com/mohammed3120/tensorflow_tutorial.git

cd tensorflow_tutorial
```
### 2. Create Virtual Environment

##### Create venv with Python 3

```bash
python3 -m venv venv
```
#### Activate the virtual environment

```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure CUDA Library Paths
1. After installing dependencies, add the following CUDA library paths to your virtual environment activation script:
```bash
cat >> venv/bin/activate << 'EOF'
```
2. paste:
```bash
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"
    export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$VENV_SITE_PACKAGES/nvidia/cublas/lib:$VENV_SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$VENV_SITE_PACKAGES/nvidia/cuda_cupti/lib:$VENV_SITE_PACKAGES/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"
fi
EOF
```

3. Re-activate the virtual environment
```bash
deactivate

source venv/bin/activate
```
### 5. Launch Jupyter Notebook
```bash
jupyter notebook
```