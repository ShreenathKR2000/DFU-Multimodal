import torch
import tensorflow as tf

print("PyTorch version:", torch.__version__)
print("CUDA available (torch):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Torch device:", torch.cuda.get_device_name(0))

print("TensorFlow version:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices('GPU'))
