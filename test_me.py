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
import progressbar

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
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

if len(sys.argv) < 3:
    print("please provide images directory as the second argument")
    raise ValueError("images directory is not provided")

model_name = sys.argv[1]
images_dir = sys.argv[2]

parent_dir = 'fine_tuned_models/' + model_name + '/'
weights_file = parent_dir + 'bottleneck_fc_model.h5'
