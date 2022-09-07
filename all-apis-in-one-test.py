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

val_folder = '/home/auro/pyt-rn18-notebook/imagewoof2-320/val/'
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

model_root_path = '/home/auro/idiom-ml-tf/rn18/'
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


def setup_recipe(model):
   """
   Should run the first convolution using im2col method, instead of kn2row.
   """
   first_conv_layer_name = None
   for layer in model.layers:
      if isinstance(layer, tf.keras.layers.Conv2D):
          first_conv_layer_name = layer.name
          break
   else:  # no break
      raise RuntimeError('cannot determine first conv layer name')

   recipe = IdiomRecipe(layer_names=[first_conv_layer_name])
   recipe.update_capability(
      first_conv_layer_name, 'conv_algorithm', None, 'im2col'
   )
   return recipe

#  new recipe
recipe = setup_recipe(imported_model)

#setup for envise eval
quant_model = setup_for_evaluation(imported_model,
                                   finetuning_method="dft",
                                   recipe=recipe)

quant_model.compile(loss='categorical_crossentropy',
              optimizer=sgd_optimizer,
              metrics=['accuracy'])

logger.info(f'Show Envise eval  accuracy')
_, _ = quant_model.evaluate(val_gen)

# reload model and apply fine-tuning
imported_model = ResnetBuilder.build_resnet_18(target_size + channels, 10)
model_path = os.path.join(model_root_path, 'checkpoint', 'rn18-best-epoch-43-acc-0.7.hdf5')
if os.path.exists(model_path):
   print(f'loading file:{model_path}')
   load_status = imported_model.load_weights(model_path)
else:
   print(f'file {model_path} does not exist.')
   exit(1)

strategy = tf.distribute.get_strategy()
recipe = IdiomRecipe()
tuned_model = setup_for_tuning(imported_model,
                               finetuning_method="dft",
                               strategy=strategy,
                               inputs=val_gen,
                               recipe=recipe)

logger.info('Done setting up for fine-tuning. Starting fine-tuning...')

sgd_optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.00001,  # carefully pick the lr as you are fine-tuning
    momentum=0.9,
    nesterov=False,
    name='SGD',
)
tuned_model.compile(loss='categorical_crossentropy',
              optimizer=sgd_optimizer,
              metrics=['accuracy'])
epochs = 1
callbacks = []
train_batch_size = 16
train_folder = '/home/auro/pyt-rn18-notebook/imagewoof2-320/train'
train_gen = image_gen.flow_from_directory(train_folder, class_mode="categorical",
                                   shuffle=True, batch_size=train_batch_size,
                                   target_size=target_size)

_ = tuned_model.fit(train_gen,
                    batch_size=train_batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    use_multiprocessing=False,)
logger.info(f'Evaluating on validation data...')
_, _  = tuned_model.evaluate(val_gen)

