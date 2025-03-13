from numba import cuda

print("CUDA available:", cuda.is_available())
if cuda.is_available():
    print("CUDA device:", cuda.get_current_device().name)