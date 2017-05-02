import sys
from keras.preprocessing.image import ImageDataGenerator

data_dir = sys.argv[1]
train_data_dir = data_dir + '/train'
val_data_dir = data_dir + '/val'

datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        zoom_range=0.2,
        fill_mode='reflect')

i = 0
for batch in datagen.flow_from_directory(train_data_dir,
                        target_size=(277, 277),
                        batch_size=1,
                        save_to_dir='preview', save_prefix='test_', save_format='jpeg'):
    i += 1
    if i > 30:
        break
