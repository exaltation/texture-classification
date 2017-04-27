from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.initializers import TruncatedNormal
from keras.models import Model
from CustomLayers.LRN import LRN2D

DROPOUT = 0.5

def create_model(num_classes):
    # first Conv2D+Relu
    x = Conv2D(96,
    input_shape=(227, 227, 3),
    kernel_size=11,
    strides=4,
    activation="relu",
    kernel_initializer=TruncatedNormal(stddev=0.01))

    # first Pooling
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # first Local response normalization
    x = LRN2D()(x)

    # second Conv2D+Relu
    x = Conv2D(256,
    kernel_size=5,
    activation="relu",
    kernel_initializer=TruncatedNormal(stddev=0.01))(x)

    # second Pooling
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # second Local response normalization
    x = LRN2D()(x)

    # third Conv2D+Relu
    x = Conv2D(384,
    kernel_size=5,
    activation="relu",
    kernel_initializer=TruncatedNormal(stddev=0.01))(x)

    # Energy layer
    x = AveragePooling2D(pool_size=13)(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    x = Dense(num_classes, activation='softmax')(x)

    return x, Input(227, 227, 3)
