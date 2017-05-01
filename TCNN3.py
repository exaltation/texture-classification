from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.initializers import TruncatedNormal, RandomNormal
from keras.models import Model
from CustomLayers.LRN import LRN2D

DROPOUT = 0.5

def create_model(num_classes):
    inputs = Input(shape=(227, 227, 3))
    # first Conv2D+Relu
    x = Conv2D(96,
    input_shape=(227, 227, 3),
    kernel_size=(11, 11),
    strides=(4, 4),
    activation="relu",
    kernel_initializer=RandomNormal(stddev=0.01),
    data_format="channels_last",
    # padding="same",
    )(inputs)

    # first Pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last")(x)

    # first Local response normalization
    # x = LRN2D()(x)

    # second Conv2D+Relu
    x = Conv2D(256,
    kernel_size=(5, 5),
    activation="relu",
    kernel_initializer=RandomNormal(stddev=0.01),
    data_format="channels_last",
    padding="same",
    )(x)

    # second Pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last")(x)

    # second Local response normalization
    # x = LRN2D()(x)

    # third Conv2D+Relu
    x = Conv2D(384,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer=RandomNormal(stddev=0.01),
    data_format="channels_last",
    padding="same",
    )(x)

    # Energy layer
    x = AveragePooling2D(pool_size=(13, 13), data_format="channels_last")(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.005))(x)
    x = Dropout(DROPOUT)(x)

    x = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.005))(x)
    x = Dropout(DROPOUT)(x)

    x = Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01))(x)

    return x, inputs
