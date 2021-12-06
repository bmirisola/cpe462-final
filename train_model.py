import numpy as np
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
#import PIL
#import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.device('/GPU:0')
