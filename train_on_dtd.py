import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from TCNN3 import create_model as TCNN3

data_dir = sys.argv[1]
train_data_dir = data_dir + '/train'
val_data_dir = data_dir + '/val'
num_classes = 47
#num_samples_per_class = 40
num_epochs = 100
batch_size = 32

xy, img_input = TCNN3(num_classes)

model = Model(inputs=img_input,
              outputs=xy)

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('tmp/TCNN3.weights.dtd.h5', save_best_only=True, verbose=1),
    EarlyStopping(monitor="loss", min_delta=0.0001, patience=15, verbose=1),
]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    data_format='channels_last')

val_datagen = ImageDataGenerator(
    rescale=1./255,
    data_format='channels_last')

train_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(227, 227),
    batch_size=batch_size)

val_dataset = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(227, 227),
    batch_size=batch_size)

model.fit_generator(
    train_dataset,
    steps_per_epoch=100*num_classes // batch_size,
    epochs=num_epochs,
    validation_data=val_dataset,
    validation_steps=20*num_classes // batch_size,
    callbacks=callbacks)
