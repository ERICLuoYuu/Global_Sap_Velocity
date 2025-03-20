from numba import cuda

print("CUDA available:", cuda.is_available())
if cuda.is_available():
    print("CUDA device:", cuda.get_current_device().name)
import tensorflow as tf
import google.protobuf
print(f"TensorFlow version: {tf.__version__}")
print(f"Protobuf version: {google.protobuf.__version__}")