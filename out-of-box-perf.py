import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from new_rn18 import ResnetBuilder
import numpy as np
import logging

# idiom.ml imports
from idiom.ml.tf import (
    setup_for_evaluation,
    setup_for_tuning,
    setup_for_export
)
from idiom.ml.tf.recipe import IdiomRecipe

logger = tf.get_logger()
logger.propagate = False
logger.setLevel(logging.INFO)
logger.info(
   f'TF version:{tf.__version__}, cuda version:{tf.sysconfig.get_build_info()["cuda_version"]}')

data_root_path = '/models/home/auro/pyt-rn18-notebook/' 
model_root_path = '/models/home/auro/bk-idiom-ml-tf/rn18/'
# data_root_path = '/home/auro/pyt-rn18-notebook/' 
# model_root_path = '/home/auro/bk-idiom-ml-tf/rn18/'

val_folder = os.path.join(data_root_path, 'imagewoof2-320/val/')
target_size = (320, 320)
channels = (3,)
nb_classes = 10
eval_batch_size = 16

image_gen = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True)
image_gen.mean = np.array([123.68, 116.779, 103.939],
                          dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
image_gen.std = 64.

val_gen = image_gen.flow_from_directory(val_folder, class_mode="categorical",
                                   shuffle=False, batch_size=eval_batch_size,
                                   target_size=target_size)

imported_model = ResnetBuilder.build_resnet_18(target_size + channels, 10)

model_path = os.path.join(model_root_path, 'checkpoint', 'rn18-best-epoch-43-acc-0.7.hdf5')
if os.path.exists(model_path):
   print(f'loading file:{model_path}')
   load_status = imported_model.load_weights(model_path)
else:
   print(f'file {model_path} does not exist.')
   exit(1)

imported_model.summary()

sgd_optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.,
    momentum=0.9,
    nesterov=False,
    name='SGD',
)
imported_model.compile(loss='categorical_crossentropy',
              optimizer=sgd_optimizer,
              metrics=['accuracy'])

logger.info(f'Show the out-of-the-box accuracy')
_, _ = imported_model.evaluate(val_gen)


