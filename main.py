import os
import time
from get_parser import get_parser

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from rn18_model import ResnetBuilder
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


def eval(model, data_loader, device):
    print("Running evaluation.")
    model.eval()
    loss = 0
    correct = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    start = time.time()
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    elapsed = int(time.time() - start)
    loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    print(
        "\n" + f"loss:          {loss:.03f}" + "\n"
        f"accuracy:      {accuracy:.01f}" + "\n"
        f"elapsed time:  {elapsed:d} sec." + "\n"
    )
    print("Evaluation complete.")


def finetune(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
):
    print(f"Training for {epochs} epochs.")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if use_adamw:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.3
    )

    for epoch in range(epochs):
        print(f"Training epoch: {epoch}")
        model.train()
        for data, target in tqdm(train_loader):
            inputs, labels = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        if not skip_train_eval:
            model.eval()
            eval(model, val_loader, device)

    print("Training complete.")


def main(
    model_path,
    data_dir,
    tune_batch_size,
    eval_batch_size,
    epochs,
    lr,
    do_oob_eval,
    do_tune,
    do_envise_eval,
    finetune_with_dft,
    finetune_with_ept,
):

    val_folder = os.path.join(data_dir, 'val')
    target_size = (320, 320)
    channels = (3,)
    nb_classes = 10

    image_gen = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
    image_gen.mean = np.array([123.68, 116.779, 103.939],
                              dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
    image_gen.std = 64.

    val_gen = image_gen.flow_from_directory(val_folder, class_mode="categorical",
                                       shuffle=False, batch_size=eval_batch_size,
                                       target_size=target_size)

    imported_model = ResnetBuilder.build_resnet_18(target_size + channels, 10)

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

    if do_oob_eval:
        logger.info(f'Show the out-of-the-box accuracy')
        _, _ = imported_model.evaluate(val_gen)


    if do_envise_eval:
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

    if do_tune:
       # reload model and apply fine-tuning
       imported_model = ResnetBuilder.build_resnet_18(target_size + channels, 10)
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
           learning_rate=lr,  # carefully pick the lr as you are fine-tuning
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
       train_folder = os.path.join(data_dir, 'train')
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

       
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(
        args.checkpoint_path,
        args.data_dir,
        args.tune_batch_size,
        args.eval_batch_size,
        args.epochs,
        args.lr,
        args.do_oob_eval,
        args.do_tune,
        args.do_envise_eval,
        args.finetune_with_dft,
        args.finetune_with_ept,
    )
