import sys
import os
import shutil

from models.tcnn import TCNN

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import numpy as np
import progressbar
from optparse import OptionParser

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data. Should have tha same structure, as the train data used in train_tcnn.py. If not provided, you will be asked to provide images for prediction.")
parser.add_option("-s", "--suffix", dest="suffix",
				help="Used to find proper weights and class names files")
parser.add_option("--batch-size", dest="batch_size",
				help="Batch size. Defaults to 16", default=16)
parser.add_option("--steps", dest="steps", help="Steps for test data. Defaults to 100", default=100)
parser.add_option("--target-size", dest="target_size",
				help="Target size to resize images to. Defaults to 227", default=227)

(options, args) = parser.parse_args()

if options.test_path:   # if test path is given
	images_dir = options.test_path

if not options.suffix:   # if suffix is not given
	parser.error('Error: suffix must be specified. Pass --suffix to command line')
suffix = options.suffix

def make_new_dir(file_path):
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

target_size = int(options.target_size)
steps = int(options.steps)

model_name = 'TCNN'
parent_dir = 'fine_tuned_models/' + model_name + '/'
weights_file = parent_dir + 'bottleneck_fc_model.'+suffix+'.h5'

if not os.path.exists(weights_file):
    raise ValueError("Cannot find weights for this model. Please, run fine_tune_me.py on it first.")

class_names = sorted(os.listdir(data_dir))
np.save(open(parent_dir + 'class_names.'+suffix+'.npy', 'w'), class_names)
num_classes = len(class_names)

model = TCNN(classes=num_classes, input_shape=(target_size, target_size, 3))
model.load_weights(weights_file)

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
                                            target_size=(target_size, target_size),
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
            target_size=(target_size, target_size),
            batch_size=int(options.batch_size))

    print("evaluating...")
    metrics = model.evaluate_generator(generator, int(options.steps))

    print("{0}: {1}".format(model.metrics_names[0], metrics[0]))
    print("{0}: {1}".format(model.metrics_names[1], metrics[1]))
