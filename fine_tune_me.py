import sys

from models.resnet50 import ResNet50
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.inception_v3 import InceptionV3
from models.xception import Xception

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
from os import path
import progressbar

model_choice = dict(resnet50=ResNet50,
                    vgg16=VGG16,
                    vgg19=VGG19,
                    inception_v3=InceptionV3,
                    xception=Xception)

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = '/home/inky/Desktop/datasets/dtd/images'

model_name = sys.argv[2]

num_classes = 47
num_epochs = 50
steps_per_epoch = 500
batch_size = 16

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='wrap',
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(277, 277),
        batch_size=batch_size)

def get_train_data():
    if path.isfile('train_features.' + model_name + '.npy') and path.isfile('train_labels.' + model_name + '.npy'):
        train_features = np.load(open('train_features.' + model_name + '.npy'))
        train_labels = np.load(open('train_labels.' + model_name + '.npy'))
        return train_features, train_labels

    model = model_choice[model_name](
        include_top=False,
        weights='imagenet',
        input_shape=(277, 277, 3),
        pooling='avg')

    print("Extracting features to train on")
    with progressbar.ProgressBar(max_value=steps_per_epoch) as bar:
        i = 0
        for batch, labels in train_generator:
            predictions = model.predict_on_batch(batch)
            if i == 0:
                train_features = predictions
                train_labels = labels
            else:
                train_features = np.concatenate([train_features, predictions])
                train_labels = np.concatenate([train_labels, labels])
            i += 1
            bar.update(i)
            if i == steps_per_epoch:
                break

    np.save(open('train_features.' + model_name + '.npy', 'w'), train_features)
    np.save(open('train_labels.' + model_name + '.npy', 'w'), train_labels)

    return train_features, train_labels

train_data, train_labels = get_train_data()

model = Sequential()
model.add(Input(shape=train_data.shape[1:]))

if model_name in ['vgg16', 'vgg19']:
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print("Start training")
model.fit(train_data, train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.15,
    callbacks=[
        ModelCheckpoint('bottleneck_fc_model.' + model_name + '.h5', save_best_only=True, verbose=2, monitor="val_acc")
    ])
