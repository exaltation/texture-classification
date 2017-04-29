import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, Adadelta
from TCNN3 import create_model as TCNN3

train_data_dir = sys.argv[1]
val_data_dir = sys.argv[2]
num_classes = 47
num_samples_per_class = 40
num_epochs = 20
batch_size = 16

xy, img_input = TCNN3(num_classes)

model = Model(inputs=img_input,
              outputs=xy)

model.compile(optimizer=Adadelta(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('tmp/TCNN3.weights.dtd.h5', save_best_only=True, verbose=1),
    EarlyStopping(monitor="loss", min_delta=0.0001, patience=10, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, verbose=1, min_lr=0.000001),
]

datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    fill_mode='nearest',
    horizontal_flip=True,
    rescale=1./255,
    data_format='channels_last')

for i in range(10):
    train_datagen = datagen.flow_from_directory(
        train_data_dir + '/' + str(i+1),
        target_size=(227, 227),
        batch_size=batch_size)

    val_datagen = datagen.flow_from_directory(
        val_data_dir + '/' + str(i+1),
        target_size=(227, 227),
        batch_size=batch_size)

    model.fit_generator(
        train_datagen,
        steps_per_epoch=num_samples_per_class*num_classes // batch_size,
        epochs=num_epochs,
        validation_data=val_datagen,
        validation_steps=num_samples_per_class*num_classes // batch_size,
        callbacks=callbacks)
