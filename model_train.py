"""
Data generation and model training methods
"""
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os
import random
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
import numpy as np


def create_generator(data_dir='',batch_size=18,target_size=(224,224),color_mode='rgb', preprocess_input='', seed=None, data_aug=True, shuffle=False):
    """
    Method used create a directory generator
    :param data_dir: Path to the input data
    :param target_size: Images size
    :param batch_size: Number of images per batch
    :param color_mode: Can be 'rgb' or 'grayscale'
    :param preprocess_input: Optional parameter for a preprocess input function to use
    :param seed: Set seed the RNG
    :param data_aug: Use or not data augmentation
    :param shuffle: Shuffle the data
    :return: Data generator

    """
    if not os.path.isdir(data_dir):
        print("Data directory not found")
        return -1

    if data_aug:
        shear_range=0.15
        height_shift_range=0.2
        width_shift_range=0.2
    else:
        shear_range=0
        height_shift_range=0
        width_shift_range=0

    datagen = ImageDataGenerator(
        shear_range=shear_range,
        height_shift_range=height_shift_range,
        width_shift_range=width_shift_range,
        preprocessing_function=preprocess_input,
        horizontal_flip=data_aug)

    gen1 = datagen.flow_from_directory(directory=data_dir, batch_size=batch_size, shuffle=shuffle, seed=seed,
                                        target_size=target_size, color_mode=color_mode,class_mode='categorical')
    return gen1

def train_model(model,
                train_samples,
                val_samples,
                train_generator,
                val_generator,
                callbacks,
                epochs=50,
                batch_size=18,
                lr=0.0001):
    """
    Method used to fit the model

    :param model: Model to be trained
    :param train_generator: Data generator for training samples
    :param val_generator:   Data generator for validation samples
    :param train_samples: Number of samples for training
    :param val_samples : Number of samples for validation
    :param callbacks:  List of callbacks used during training stage
    :param epochs:  Number of epochs for training the model
    :param batch_size: Number of images per batch
    :param lr: Learning rate for the model
    :return: history object where some performance metrics are stored
    """

    nb_train_samples = np.ceil(train_samples/batch_size)
    if val_generator is not None:
        nb_validation_samples = np.ceil(val_samples/batch_size)
    else:
        nb_validation_samples = None

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.adam(lr=lr),
                  metrics=['acc'])

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=nb_validation_samples,
        verbose=2,
        max_queue_size=1,
        shuffle=False,
        workers=1,
        use_multiprocessing=False)

    return history