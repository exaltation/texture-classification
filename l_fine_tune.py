import sys
from models.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from os import path

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = '/home/inky/Desktop/datasets/dtd/images'

num_classes = 47
num_epochs = 200
steps_per_epoch = 500
batch_size = 32

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='constant',
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(277, 277),
        batch_size=batch_size)

def get_train_data():
    if path.isfile('train_features.resnet50.npy') and path.isfile('train_labels.resnet50.npy'):
        train_features = np.load(open('train_features.resnet50.npy'))
        train_labels = np.load(open('train_labels.resnet50.npy'))
        return train_features, train_labels

    model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(277, 277, 3))

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
        if i == steps_per_epoch:
            break

    np.save(open('train_features.resnet50.npy', 'w'), train_features)
    np.save(open('train_labels.resnet50.npy', 'w'), train_labels)

    return train_features, train_labels

train_data, train_labels = get_train_data()

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=num_epochs,
          batch_size=batch_size,
          validation_split=0.15)

model.save_weights('bottleneck_fc_model.resnet50.h5')
