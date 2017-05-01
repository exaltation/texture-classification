import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from TCNN3 import create_model as TCNN3

train_data_dir = sys.argv[1]
val_data_dir = sys.argv[2]
num_classes = 47
#num_samples_per_class = 40
num_epochs = 100
batch_size = 32

xy, img_input = TCNN3(num_classes)

model = Model(inputs=img_input,
              outputs=xy)

model.compile(optimizer=Adam(lr=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('tmp/TCNN3.weights.dtd.h5', save_best_only=True, verbose=1),
    # EarlyStopping(monitor="loss", min_delta=0.00001, patience=15, verbose=1),
    # ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1, min_lr=0.000001),
]

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    rescale=1./255,
    data_format='channels_last')

val_datagen = ImageDataGenerator(
    fill_mode='reflect',
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
    steps_per_epoch=102*num_classes // batch_size,
    epochs=num_epochs,
    validation_data=val_dataset,
    validation_steps=18*num_classes // batch_size,
    callbacks=callbacks)
