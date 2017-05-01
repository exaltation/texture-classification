import sys
from models.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout

train_data_dir = sys.argv[1]
val_data_dir = sys.argv[2]
num_classes = 47
num_epochs = 50
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    samplewise_center=True,
    samplewise_std_normalization=True)

val_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(277, 277),
    batch_size=batch_size)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(277, 277),
    batch_size=batch_size)

model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(227, 227, 3))

top_model = Flatten(input_shape=model.output_shape[1:])
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes, activation='softmax')(top_model)
# top_model = Sequential()
# top_model.add(Flatten())
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(num_classes, activation='softmax'))

model = model(top_model)

for layer in model.layers[:-3]:
    print(layer.name)
    layer.trainable = False
