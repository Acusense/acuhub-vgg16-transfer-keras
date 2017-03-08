import os, math
import numpy as np
import json
import keras as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils

data_path = os.path.join(os.environ['BASE_PATH'], 'data')
config_path = os.path.join(os.environ['BASE_PATH'], 'config.json')

data_config = json.load(open(config_path))['data']

samples = data_config['file_tags_map']
nb_classes = max(max([sample[1] for sample in samples])) + 1
nb_samples = len(samples)
nb_train_samples = int(math.floor(nb_samples * data_config['train_val_split']))
nb_val_samples = nb_samples - nb_train_samples

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(**data_config['train_preprocess'])

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(**data_config['test_preprocess'])

# craft data and labels from files
if "dim_ordering" in data_config['train_preprocess'].keys():
    dim_ordering = data_config['train_preprocess']['dim_ordering']
else:
    dim_ordering = K.image_dim_ordering()

if dim_ordering == "tf":
    X_train = np.zeros((nb_train_samples, data_config['width'],
                       data_config['height'], 3))
    X_val = np.zeros((nb_val_samples, data_config['width'],
                     data_config['height'], 3))
else:
    X_train = np.zeros((nb_train_samples, 3, data_config['width'],
                       data_config['height']))

    X_val = np.zeros((nb_val_samples, 3, data_config['width'],
                     data_config['height']))
y_train = []
y_val = []

np.random.shuffle(samples)

target_size = (data_config['height'], data_config['width'])
for ind, sample in enumerate(samples):
    img_path = os.path.join(data_path, sample[0])
    if ind in range(0, nb_train_samples):
        X_train[ind] = img_to_array(load_img(img_path, target_size=target_size))
        y_train.append(sample[1])
    elif ind in range(nb_train_samples, len(samples)):
        X_val[ind - nb_train_samples] = img_to_array(load_img(img_path, target_size=target_size))
        y_val.append(sample[1])

# Convert y to categorical variables
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)

train_generator = train_datagen.flow(
    X_train,
    Y_train,
    batch_size=data_config['batch_size']
)

val_generator = test_datagen.flow(
    X_val,
    Y_val,
    batch_size=data_config['batch_size']
)


# train_generator = train_datagen.flow(
#         train_data_dir,
#         target_size=(data_config['width'], data_config['height']),
#         batch_size=data_config['batch_size'],
#         class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#         validation_data_dir,
#         target_size=(data_config['width'], data_config['height']),
#         batch_size=data_config['batch_size'],
#         class_mode='categorical')


