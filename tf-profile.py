import os
import time
import argparse

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import logging
import itertools

from idiom.runtime import profile

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compile the ONNX model for Envise')
    parser.add_argument('--data-dir',
                        type=str, 
                        help='Path to ResNet18 dataset', required=True)
    parser.add_argument('--compiled-model-dir',
                        type=str, 
                        help='Path to compiled output', required=True)
    args = parser.parse_args()

    return args


# data_dir = '../../00-getting-started/imagewoof2-320'
# data_dir = '/idiom-eap/examples/00-getting-started/imagewoof2-320'
# compiled_model_dir = './compile_dir'
args = parse_arguments()
val_folder = os.path.join(args.data_dir, 'val')
# train_folder = os.path.join(args.data_dir, 'train')
target_size = (320, 320)
channels = (3,)
nb_classes = 10
eval_batch_size = 1

image_gen = ImageDataGenerator(featurewise_center=True,
                               featurewise_std_normalization=True)
image_gen.mean = np.array([123.68, 116.779, 103.939],
                          dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
image_gen.std = 64.
# val_folder = os.path.join(data_dir, 'val')
val_gen = image_gen.flow_from_directory(val_folder, class_mode="categorical",
                                        shuffle=False, batch_size=eval_batch_size,
                                        target_size=target_size)
x_batch, y_batch = next(val_gen)
print(f'x shape: {x_batch.shape} y shape: {y_batch.shape}')
data = [{'input': np.expand_dims(x, axis=0) for x in itertools.islice(x_batch, eval_batch_size)}]

profile(args.compiled_model_dir, data, detailed_report=True)
