import sys
from models.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD

train_data_dir = sys.argv[1]
val_data_dir = sys.argv[2]
num_classes = 47
num_epochs = 50
batch_size = 16
#====================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    samplewise_center=True,
    samplewise_std_normalization=True)

val_datagen = ImageDataGenerator(
    rescale=1./255)
#====================================================================
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=batch_size)
#====================================================================
model = ResNet50(
    include_top=False,
    weights='imagenet')
#====================================================================
top_model = Flatten()(model.output)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes, activation='softmax')(top_model)
#====================================================================
main_model = Model(inputs=model.input, outputs=top_model)

for layer in main_model.layers[:-6]:
    layer.trainable = False
#====================================================================
main_model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy'])
#====================================================================
main_model.fit_generator(
    train_generator,
    steps_per_epoch=100*num_classes // batch_size,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=20*num_classes // batch_size)
