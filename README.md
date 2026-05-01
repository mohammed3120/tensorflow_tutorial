# TensorFlow GPU/CPU Device Management Demo

This Jupyter notebook demonstrates how to manage and test GPU/CPU device placement in TensorFlow 2.x, including device assignment, verification, and basic tensor operations.

## 📋 System Requirements

- **OS**: Ubuntu 22.04 LTS (or compatible Linux distribution)
- **Python**: 3.10.x (3.10.12 specifically tested)
- **TensorFlow**: 2.21.0
- **CUDA**: Required for GPU support (optional, falls back to CPU)
- **Jupyter**: For running the notebook

## 📑 Contents

### 00_gpu_cpu.ipynb
**GPU/CPU Device Management** - Demonstrates device placement and GPU/CPU configuration in TensorFlow 2.x.

- Device placement with `tf.device()`
- Checking TensorFlow version and CUDA support
- Listing available physical and logical devices
- Testing GPU operations with matrix multiplication
- Verifying tensor device assignments
- Setting environment variables for TensorFlow logging

---

### 01_Basics.ipynb
**Tensor Fundamentals** - Introduction to creating and manipulating tensors of different dimensions.

- **0D to 4D tensors**: Creating scalars, vectors, matrices, and higher-dimensional tensors
- **Casting**: Converting between tensor data types (`tf.cast`)
- **NumPy conversion**: Converting NumPy arrays to tensors (`tf.convert_to_tensor`)
- **Identity matrices**: Creating identity matrices with `tf.eye()`
- **Fill operations**: Creating tensors with constant values (`tf.fill`)
- **Ones and Zeros**: `tf.ones()`, `tf.zeros()`, `tf.ones_like()`, `tf.zeros_like()`
- **Tensor properties**: Understanding `rank` (number of dimensions) and `size` (total elements)
- **Random tensors**: Generating random values with `tf.random.normal()` (normal distribution) and `tf.random.uniform()` (uniform distribution)
- **Random seeds**: Controlling reproducibility with `tf.random.set_seed()`

---

### 02_Indexing_MathOperations.ipynb
**Indexing and Mathematical Operations** - Comprehensive guide to accessing tensor elements and performing mathematical operations.

**Indexing Operations:**
- Basic indexing (single element, negative indices)
- Slicing (start:stop:step)
- Ellipsis (`...`) for multiple dimensions

**Mathematical Operations:**
- `tf.abs()` - Absolute value (including complex numbers)
- `tf.acos()` - Inverse cosine
- `tf.add()` - Element-wise addition with broadcasting
- `tf.add_n()` - Sum of multiple tensors
- `tf.argmax()` - Index of maximum value along axis
- `tf.ceil()` - Ceiling (round up)
- `tf.floor()` - Floor (round down)
- `tf.floordiv()` - Integer division
- `tf.multiply()` - Element-wise multiplication with broadcasting
- `tf.math.confusion_matrix()` - Create confusion matrix
- `tf.math.count_nonzero()` - Count non-zero elements
- `tf.math.equal()` - Element-wise equality comparison
- `tf.math.exp()` - Exponential (e^x)
- `tf.math.greater()` / `greater_equal()` - Comparison operations
- `tf.math.maximum()` / `minimum()` - Element-wise max/min
- `tf.math.negative()` - Negation
- `tf.reduce_max()` / `reduce_min()` - Reduce operations
- `tf.math.sigmoid()` - Sigmoid activation function
- `tf.math.top_k()` - Top K values and indices

---

### 03_LinearAlgebraOperations.ipynb
**Linear Algebra Operations** - Matrix operations and linear algebra computations.

- **`tf.linalg.matmul()` / `@` operator**: Matrix multiplication for 2D, 3D, and 4D tensors
  - Batch matrix multiplication
  - Broadcasting rules
  - Transpose options

- **`tf.linalg.band_part()`**: Extract banded matrix (keep diagonals and sub/super-diagonals)

- **`tf.linalg.inv()`**: Matrix inverse

- **`tf.linalg.det()`**: Matrix determinant

- **`tf.linalg.svd()`**: Singular Value Decomposition
  - U: Left singular vectors
  - S: Singular values (diagonal)
  - V: Right singular vectors
  - Reconstruction: `mat = U @ tf.linalg.diag(S) @ tf.transpose(V)`

- **`tf.transpose()`**: Transpose tensors of any dimension

---

### 04_Common_Tensorflow_Functions.ipynb
**Common Tensor Manipulation Functions** - Essential utility functions for tensor reshaping and manipulation.

- **`tf.expand_dims()`**: Add new dimensions (increase rank)
  - Axis control (0, 1, 2, -1)
  - Applications for broadcasting

- **`tf.squeeze()`**: Remove dimensions of size 1
  - Selective axis removal

- **`tf.reshape()`**: Change tensor shape
  - Using `-1` for inferred dimensions

- **`tf.concat()`**: Concatenate tensors along existing axis
  - Row-wise (axis=0) and column-wise (axis=1) concatenation

- **`tf.stack()`**: Stack tensors along new axis (increases rank)
  - Difference between `stack` and `concat`

- **`tf.pad()`**: Pad tensors
  - Padding modes: `CONSTANT`, `REFLECT`, `SYMMETRIC`
  - Custom padding values

- **`tf.gather()`**: Gather slices from tensor along axis
  - Index-based selection

- **`tf.gather_nd()`**: Gather slices using multi-dimensional indices
  - Nested indexing

---

### 05_ragged_sparse_string_varibles.ipynb
**Special Data Types** - Ragged tensors, sparse tensors, strings, and variables.

**Ragged Tensors (tf.RaggedTensor):**
- `tf.ragged.constant()` - Create ragged tensors (variable-length rows)
- `tf.ragged.boolean_mask()` - Mask ragged tensors
- `tf.RaggedTensor.from_row_lengths()` - Create from row lengths
- `tf.RaggedTensor.from_row_limits()` - Create from row limits
- `tf.RaggedTensor.from_row_starts()` - Create from row starts
- `tf.RaggedTensor.from_tensor()` - Convert dense to ragged

**Sparse Tensors (tf.sparse.SparseTensor):**
- Create sparse tensors with indices, values, and dense shape
- `tf.sparse.to_dense()` - Convert sparse to dense

**String Operations:**
- `tf.strings.join()` - Concatenate strings

**Variables (tf.Variable):**
- Mutable tensors for trainable parameters
- `assign()` method for value updates
- Index assignment (e.g., `var[0].assign(new_value)`)
- Comparison: `tf.constant` (immutable) vs `tf.Variable` (mutable)

---

## 🎯 Learning Path

| Notebook | Prerequisites | Key Concepts |
|----------|--------------|--------------|
| 00_gpu_cpu | None | Device management, GPU/CPU |
| 01_Basics | 00_gpu_cpu | Tensor creation, shapes, dtypes |
| 02_Indexing_MathOperations | 01_Basics | Indexing, mathematical operations |
| 03_LinearAlgebraOperations | 01_Basics, 02 | Matrix operations, SVD |
| 04_Common_Functions | 01_Basics | Reshaping, padding, gathering |
| 05_Special_Types | 01_Basics | Ragged, sparse, variables |

---

## 📊 Quick Reference Table

| Operation Type | Key Functions |
|----------------|---------------|
| **Creation** | `tf.constant()`, `tf.eye()`, `tf.fill()`, `tf.ones()`, `tf.zeros()` |
| **Random** | `tf.random.normal()`, `tf.random.uniform()`, `tf.random.set_seed()` |
| **Reshape** | `tf.reshape()`, `tf.expand_dims()`, `tf.squeeze()` |
| **Combine** | `tf.concat()`, `tf.stack()` |
| **Index/Gather** | `[]` indexing, `tf.gather()`, `tf.gather_nd()` |
| **Math** | `tf.add()`, `tf.multiply()`, `tf.abs()`, `tf.exp()` |
| **Linear Algebra** | `@` (matmul), `tf.linalg.inv()`, `tf.linalg.svd()` |
| **Reduction** | `tf.reduce_max()`, `tf.reduce_min()`, `tf.argmax()` |
| **Special** | `tf.RaggedTensor`, `tf.sparse.SparseTensor`, `tf.Variable` |

---

## 🔍 Notebook Details

### 01_Basics - Tensor Shapes Examples
- **Rank 0**: Scalar `tf.constant(5)`
- **Rank 1**: Vector `tf.constant([1,2,3])`
- **Rank 2**: Matrix `tf.constant([[1,2],[3,4]])`
- **Rank 3**: 3D tensor `shape=(4,2,3)`
- **Rank 4**: 4D tensor `shape=(2,4,2,3)`

### 03_LinearAlgebraOperations - Multi-dimensional Matmul
- **2D**: `(m,n) @ (n,p) = (m,p)`
- **3D**: `(b,m,n) @ (b,n,p) = (b,m,p)`
- **4D**: `(b,c,m,n) @ (b,c,n,p) = (b,c,m,p)`

### 05_Special_Types - Key Differences
- **Tensor (`tf.constant`)**: Immutable, faster for fixed data
- **Variable (`tf.Variable`)**: Mutable, trainable parameters
- **Ragged**: Variable-length sequences
- **Sparse**: Memory-efficient for mostly-zero data

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