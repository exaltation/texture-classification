# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten

def TCNN(input_shape=(277, 277, 3), classes=47):
    img_input = Input(shape=input_shape)

    # block 1
    x = Conv2D(64, (7, 7), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 2
    x = Conv2D(128, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 3
    x = Conv2D(256, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # block 4
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # energy pooling
    x = GlobalAveragePooling2D()(x)

    # fully-connected
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x, name='tcnn')

    return model

def TCNN2(input_shape=(277, 277, 3), classes=47):
    img_input = Input(shape=input_shape)

    # block 1
    x = Conv2D(66, (3, 3), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 2
    x = Conv2D(121, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 3
    x = Conv2D(176, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 4
    x = Conv2D(220, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # energy pooling
    x = GlobalAveragePooling2D()(x)

    # fully-connected
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x, name='tcnn2')

    return model
