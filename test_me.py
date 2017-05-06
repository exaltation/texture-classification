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
from optparse import OptionParser

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data. Should have tha same structure, as the train data used in fine_tune_me.py. If not provided, you will be asked to provide images for prediction.")
parser.add_option("-m", "--model", dest="model_name", help="Specify a model for test: resnet50, vgg16, vgg19, inception_v3 or xception (model should be already fine-tuned using fine_tune_me.py).")
parser.add_option("-s", "--suffix", dest="suffix", help="Used to find proper weights and class names files")
parser.add_option("--batch-size", dest="batch_size", help="Batch size for test data. Defaults to 16", default=16)
parser.add_option("--steps", dest="steps", help="Steps for test data. Defaults to 100", default=100)

(options, args) = parser.parse_args()

if options.test_path:   # if test path is given
	images_dir = options.test_path

if not options.model_name:   # if model name is not given
	parser.error('Error: model name must be specified. Pass --model to command line')
model_name = options.model_name

if not options.suffix:   # if suffix is not given
	parser.error('Error: suffix must be specified. Pass --suffix to command line')
suffix = options.suffix

def make_new_dir(file_path):
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

if model_name not in ['resnet50', 'vgg16', 'vgg19', 'inception_v3', 'xception']:
    print("please choose one of the following")
    print("resnet50, vgg16, vgg19, inception_v3, xception")
    raise ValueError("Model name is invalid")

model_choice = dict(resnet50=ResNet50,
                    vgg16=VGG16,
                    vgg19=VGG19,
                    inception_v3=InceptionV3,
                    xception=Xception)

parent_dir = 'fine_tuned_models/' + model_name + '/'
weights_file = parent_dir + 'bottleneck_fc_model.'+suffix+'.h5'

class_names = np.load(open(parent_dir + 'class_names.'+suffix+'.npy'))
num_classes = len(class_names)

if not os.path.exists(weights_file):
    raise ValueError("Cannot find weights for this model. Please, run fine_tune_me.py on it first.")

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

if not options.test_path:
    my_root = os.path.dirname(os.path.realpath(__file__))
    print("Please, enter path to folder, which contains images to evaluate on.")
    files_path = raw_input()
    while not files_path.endswith('/') or not os.path.exists(files_path):
		if not os.path.exists(files_path):
			print("Path doesn't exist, try once more")
			files_path = raw_input()
		else:
			print("Path should end with a slash ('/'), try once more")
			files_path = raw_input()


    files = []
    for (_, _, filenames) in os.walk(files_path):
        files.extend(filenames)
        break

    f_count = 0
	_filenames = []
    for f in files:
        if not f.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            print("File {0} has wrong extension".format(f))
            print("supported file extensions are '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'")
            continue

        d = my_root + '/temporary/'+os.path.splitext(f)[0]+'/'
        make_new_dir(d)
        shutil.copyfile(files_path + f, d + f)
        f_count += 1
		_filenames.append(f)

	_filenames = sorted(_filenames)
    for batch, lbls in datagen.flow_from_directory(my_root + '/temporary/',
                                            target_size=(277, 277),
                                            batch_size=1):
        prediction = model.predict_on_batch(batch)
        print('file {0}: {1}'.format(_filenames[lbls.argmax()], class_names[prediction.argmax()]))
        f_count -= 1
        if f_count == 0:
            break

    shutil.rmtree(my_root + '/temporary/')
else:
    print("plotting model...")
    plot_model(model, show_shapes=True, to_file=os.path.dirname(weights_file) + '/model.png')

    generator = datagen.flow_from_directory(
            images_dir,
            target_size=(277, 277),
            batch_size=options.batch_size)

    print("evaluating...")
    metrics = model.evaluate_generator(generator, options.steps)

    print("{0}: {1}".format(model.metrics_names[0], metrics[0]))
    print("{0}: {1}".format(model.metrics_names[1], metrics[1]))
