import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from TCNN3 import create_model as TCNN3

train_data_dir = sys.argv[1]
val_data_dir = sys.argv[2]
num_samples = 47*40
batch_size = 10
xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = TCNN3.create_model()

model = Model(inputs=[img_input],
              outputs=[xy])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    # samplewise_center=True,
    # samplewise_std_normalization=True,
    fill_mode='nearest',
    horizontal_flip=True,
    rescale=1./255,
    data_format='channels_last')

train_datagen = datagen.flow_from_directory(
    train_data_dir,
    target_size=(227, 227),
    batch_size=batch_size)

val_datagen = datagen.flow_from_directory(
    val_data_dir,
    target_size=(227, 227),
    batch_size=batch_size)

callbacks = [
    ModelCheckpoint('tmp/TCNN3.weights.dtd.h5', save_best_only=True, verbose=1),
    EarlyStopping(min_delta=0.001, patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.2, patience=1, verbose=1, min_lr=0.00005)
]

model.fit_generator(
    train_datagen,
    steps_per_epoch=num_samples // batch_size,
    epochs=50,
    validation_data=val_datagen,
    validation_steps=num_samples // batch_size,
    callbacks=callbacks)
