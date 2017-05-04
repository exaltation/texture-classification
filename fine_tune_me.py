import sys
import os

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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

model_choice = dict(resnet50=ResNet50,
                    vgg16=VGG16,
                    vgg19=VGG19,
                    inception_v3=InceptionV3,
                    xception=Xception)

model_name = sys.argv[1]
if len(sys.argv) > 2:
    data_dir = sys.argv[2]
else:
    data_dir = '/home/inky/Desktop/datasets/dtd/images'

num_classes = 47
num_epochs = 50
steps_per_epoch = 500
batch_size = 16

parent_dir = 'fine_tuning_models/' + model_name + '/'
ensure_dir(parent_dir)
features_file = parent_dir + 'train_features.npy'
labels_file = parent_dir + 'train_labels.npy'
weights_file = parent_dir + 'bottleneck_fc_model.h5'

def get_train_data():
    if path.isfile(features_file) and path.isfile(labels_file):
        train_features = np.load(open(features_file))
        train_labels = np.load(open(labels_file))
        return train_features, train_labels

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

    np.save(open(features_file, 'w'), train_features)
    np.save(open(labels_file, 'w'), train_labels)

    return train_features, train_labels

train_data, train_labels = get_train_data()

model = Sequential()
if model_name in ['vgg16', 'vgg19']:
    model.add(Dense(4096, activation='relu', input_shape=train_data.shape[1:]))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
else:
    model.add(Dense(num_classes, activation='softmax', input_shape=train_data.shape[1:]))

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
        ModelCheckpoint(weights_file, save_best_only=True, verbose=2, monitor="val_acc")
    ])
