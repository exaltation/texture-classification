import sys
import os
import shutil

from models.resnet50 import ResNet50
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.inception_v3 import InceptionV3
from models.xception import Xception

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

import numpy as np
import progressbar

num_classes = 47

def make_new_dir(file_path):
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

if len(sys.argv) < 2 or sys.argv[1] not in ['resnet50', 'vgg16', 'vgg19', 'inception_v3', 'xception']:
    print("please choose one of the following")
    print("resnet50, vgg16, vgg19, inception_v3, xception")
    raise ValueError("model name is invalid or not provided")

model_choice = dict(resnet50=ResNet50,
                    vgg16=VGG16,
                    vgg19=VGG19,
                    inception_v3=InceptionV3,
                    xception=Xception)

if len(sys.argv) > 2:
    images_dir = sys.argv[2]
else:
    images_dir = '/home/inky/Desktop/datasets/dtd/images'

model_name = sys.argv[1]

parent_dir = 'fine_tuned_models/' + model_name + '/'
weights_file = parent_dir + 'bottleneck_fc_model.h5'

if not os.path.exists(weights_file):
    raise ValueError("cannot find weights for this model")

notop_model = model_choice[model_name](include_top=False,
                                    weights='imagenet',
                                    input_shape=(277, 277, 3),
                                    pooling='avg')

top_model = Sequential()
if model_name in ['vgg16', 'vgg19', 'resnet50']:
    top_model.add(Dense(4096, activation='relu', input_shape=notop_model.output_shape[1:]))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dense(num_classes, activation='softmax'))
else:
    top_model.add(Dense(num_classes, activation='softmax', input_shape=notop_model.output_shape[1:]))

top_model.load_weights(weights_file)

model = Sequential()
model.add(notop_model)
model.add(top_model)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

datagen = ImageDataGenerator(
    rescale=1./255)

my_root = '/home/inky/Documents/texture-classification'

print("do you wish to test it on your own images? (y/N)")
ans = raw_input()
if ans == 'y' or ans == 'Y':
    class_names = sorted(os.listdir("/home/inky/Desktop/datasets/dtd/images"))

    while True:
        print("enter image file path")
        file_path = raw_input().rstrip()
        if not os.path.isfile(file_path) or not file_path.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            print("File doesn't exist or has wrong extension")
            print("supported file extensions are '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'")
            continue


        make_new_dir(my_root + '/temporary/t/')
        shutil.copyfile(file_path, my_root + '/temporary/t/' + file_path.split('/')[-1])
        for batch, lbls in datagen.flow_from_directory(my_root + '/temporary/',
                                                target_size=(277, 277),
                                                batch_size=1):
            print(batch.shape)
            prediction = model.predict_on_batch(batch)
            break

        print(class_names[prediction.argmax()])

        print("wish one more? (y/N)")
        ans = raw_input()
        if ans != 'y' and ans != 'Y':
            break
else:
    print("plotting model...")
    plot_model(model, show_shapes=True, to_file=os.path.dirname(weights_file) + '/model.png')

    generator = datagen.flow_from_directory(
            images_dir,
            target_size=(277, 277),
            batch_size=16)

    print("evaluating...")
    metrics = model.evaluate_generator(generator, 100)

    print("{0}: {1}".format(model.metrics_names[0], metrics[0]))
    print("{0}: {1}".format(model.metrics_names[1], metrics[1]))
