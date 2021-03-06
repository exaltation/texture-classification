import sys
import os

from models.tcnn import TCNN, TCNN2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import progressbar
from optparse import OptionParser

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-v", "--val", dest="val_path", help="Path to validation data.")
parser.add_option("-m", "--model", dest="model_name", help="Model name. Defaults to TCNN", default="TCNN")
parser.add_option("-s", "--suffix", dest="suffix",
				help="Model will be saved with provided suffix, i.e. bottleneck_fc_model.[suffix].h5.")
parser.add_option("--num-epochs", dest="num_epochs", help="Number of epochs. Defaults to 100", default=100)
parser.add_option("--batch-size", dest="batch_size",
				help="Batch size. Defaults to 16", default=16)
parser.add_option("--steps-per-epoch", dest="steps_per_epoch",
				help="Steps per epoch. Defaults to 100", default=300)
parser.add_option("--validation-steps", dest="validation_steps",
				help="Validation steps. Defaults to 14", default=60)
parser.add_option("--target-size", dest="target_size",
				help="Target size to resize images to. Defaults to 227", default=227)
parser.add_option("--optimizer", dest="optimizer",
				help="Optimizer to train the model. Supported values: adam, nadam, adagrad, adadelta, adamax. Defaults to adam. See keras.io/optimizers for more details.",
				default='adam')
parser.add_option("--continue",
                  action="store_true", dest="_continue", default=False,
                  help="Load lately saved weights and continue training")

(options, args) = parser.parse_args()

if options.optimizer not in ('adam', 'nadam', 'adagrad', 'adadelta', 'adamax'):
	parser.error('Error: Supported values for the optimizer: adam, nadam, adagrad, adadelta, adamax. Given {0}'.format(options.optimizer))

if not options.train_path:   # if train path is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')
data_dir = options.train_path

if not options.val_path:   # if train path is not given
	parser.error('Error: path to validation data must be specified. Pass --val to command line')
val_dir = options.val_path

if not options.suffix:   # if suffix is not given
	parser.error('Error: suffix must be specified. Pass --suffix to command line')
suffix = options.suffix

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

target_size = int(options.target_size)
num_epochs = int(options.num_epochs)
steps_per_epoch = int(options.steps_per_epoch)
validation_steps = int(options.validation_steps)
batch_size = int(options.batch_size)

model_choice = dict(TCNN=TCNN, TCNN2=TCNN2)

model_name = options.model_name
if model_name not in ('TCNN', 'TCNN2'):
	parser.error('Error: Model name should be TCNN or TCNN2')
parent_dir = 'fine_tuned_models/' + model_name + '/'
ensure_dir(parent_dir)
weights_file = parent_dir + 'bottleneck_fc_model.'+suffix+'.h5'

class_names = sorted(os.listdir(data_dir))
np.save(open(parent_dir + 'class_names.'+suffix+'.npy', 'w'), class_names)
num_classes = len(class_names)

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(target_size, target_size),
    batch_size=batch_size)

val_datagen = ImageDataGenerator(
    rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(target_size, target_size),
    batch_size=batch_size)

model = model_choice[model_name](classes=num_classes, input_shape=(target_size, target_size, 3))

if options._continue:
	if not os.path.exists(weights_file):
	    print("Cannot find weights for this model. Please, run fine_tune_me.py on it first.")
	else:
	    model.load_weights(weights_file)

model.compile(
    loss='categorical_crossentropy',
    optimizer=options.optimizer,
    metrics=['accuracy'])

print("Start training")
model.fit_generator(train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[
        ModelCheckpoint(weights_file, save_best_only=True, verbose=2, monitor='val_acc')
    ])
