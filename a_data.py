## DATA INPUT
import os
from keras.preprocessing.image import ImageDataGenerator

# Read from file
data_path = '/data/cat_dog_train'

img_width, img_height = 150, 150

train_data_dir = os.path.join(data_path, 'train')
validation_data_dir = os.path.join(data_path, 'validation')
nb_train_samples = 2000
nb_validation_samples = 800




# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')


