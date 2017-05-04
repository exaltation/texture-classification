import sys
from models.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from os import path
import progressbar

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = '/home/inky/Desktop/datasets/dtd/images'

num_classes = 47
num_epochs = 40
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
    if path.isfile('train_features.xception.npy') and path.isfile('train_labels.xception.npy'):
        train_features = np.load(open('train_features.xception.npy'))
        train_labels = np.load(open('train_labels.xception.npy'))
        return train_features, train_labels

    model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(277, 277, 3),
        pooling='avg')
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

    np.save(open('train_features.xception.npy', 'w'), train_features)
    np.save(open('train_labels.xception.npy', 'w'), train_labels)

    return train_features, train_labels

train_data, train_labels = get_train_data()

model = Sequential()
# model.add(Dense(2048, activation='relu'))
model.add(Dense(num_classes, activation='softmax', input_shape=train_data.shape[1:]))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_data, train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.15,
    callbacks=[
        ModelCheckpoint('bottleneck_fc_model.xception.h5', save_best_only=True, verbose=2, monitor="val_acc")
    ])
