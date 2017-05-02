import sys
from models.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

data_dir = sys.argv[1]
train_data_dir = data_dir + '/train'
val_data_dir = data_dir + '/val'
num_classes = 47
num_epochs = 300
batch_size = 32

weights_path = 'tmp/resnet50.dtd_finetune.74.h5'
#====================================================================
train_datagen = ImageDataGenerator(
    zoom_range=0.25,
    rescale=1./255)

val_datagen = ImageDataGenerator(
    rescale=1./255)
#====================================================================
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(277, 277),
    batch_size=batch_size)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(277, 277),
    batch_size=batch_size)
#====================================================================
model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(277, 277, 3))
#====================================================================
top_model = Flatten()(model.output)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
#top_model = Dense(4096, activation='relu')(top_model)
#top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes, activation='softmax')(top_model)
#====================================================================
main_model = Model(inputs=model.input, outputs=top_model)
main_model.load_weights(weights_path)
for layer in main_model.layers[:-4]:
    layer.trainable = False
#====================================================================
main_model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=1e-4, momentum=0.9, nesterov=True),
    metrics=['accuracy'])
#====================================================================
callbacks = [
    ModelCheckpoint('tmp/resnet50.dtd_finetune.h5', save_best_only=True, verbose=1),
    EarlyStopping(monitor="loss", min_delta=0.0001, patience=10, verbose=1)
]
#====================================================================
main_model.fit_generator(
    train_generator,
    steps_per_epoch=100*num_classes // batch_size,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=20*num_classes // batch_size,
    callbacks=callbacks)
