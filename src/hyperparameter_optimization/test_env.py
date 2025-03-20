import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_gpu_available())

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())