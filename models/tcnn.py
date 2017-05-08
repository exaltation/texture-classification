# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D

def TCNN(input_shape=(277, 277, 3), classes=47):
    img_input = Input(shape=input_shape)

    # block 1
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 2
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x, name='tcnn')

    return model
