"""
Methods for generating CNN structures
"""

from keras import applications
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model


def generate_model_VGG16(weights_path='imagenet',
                   input_shape=(224,224,3),
                   n_dense_layer=1,
                   drop_value=0.2,
                   dense_neurons=256,
                   out_neurons=5,
                   activation='softmax',
                   fine_tuning=True,
                   include_top=True):

    """
    Generate the CNN layer structure based on pre-trained Keras CNN VGG16

    :param weights_path: Optionally can load weights from stored file. Otherwise it will be downloaded from ImageNet
    :param input_shape: Input shape for the model
    :param n_dense_layer: Number of dense and dropout layer to be included at the top of the CNN
    :param drop_value: Number between 0 and 1. Fraction for the input units to drop
    :param dense_neurons: Number of units in dense layers. If several dense layers are included, the following layers units number will be decreased by *DenseNeurons/2^ly* where *ly* is the actual layer position.
    :param out_neurons: Number of units in output layer
    :param activation:  Activation function for the last layer
    :param fine_tuning: Whether or not freeze VGG16 first layers
    :param include_top: Wheter or not include last classification layer
    :return model
    """

    base_model = applications.VGG16(weights=weights_path, include_top=not(fine_tuning), input_shape=input_shape)

    x = base_model.output
    x = Flatten()(x)
    for i in range(n_dense_layer):
        x = Dropout(drop_value)(x)
        x = Dense(int(dense_neurons / 2 ** i), activation='relu')(x)

    if include_top:
        x = (Dense(out_neurons, activation=activation))(x)

    model=Model(inputs=base_model.inputs,outputs=x)

    if fine_tuning:
        for layer in base_model.layers:
            layer.trainable = False
    print('VGG16 model loaded.')

    return model
