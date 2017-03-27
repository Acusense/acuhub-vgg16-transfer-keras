import os, json, math
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import np_utils
from keras_util import ImageDataGeneratorAcusense
from __init__ import config_dict
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
K.get_session().run(tf.global_variables_initializer())

data_path = os.environ['DATA_DIR']
data_file_tags_path = os.path.join(data_path, 'data.json')

data_config = config_dict['data_config']
data_file_tags = json.load(open(data_file_tags_path))

samples = data_file_tags['file_tags_map']
tag_objects = data_file_tags['tags']
nb_classes = max(max([sample[1] for sample in samples])) + 1
nb_samples = len(samples)
nb_train_samples = int(math.floor(nb_samples * data_config['train_val_split']))
nb_val_samples = nb_samples - nb_train_samples
use_generator = data_config['use_generator']

with tf.device(os.environ['TENSORFLOW_DEVICE']):
    # craft data and labels from files
    if "dim_ordering" in data_config['train_preprocess'].keys():
        dim_ordering = data_config['train_preprocess']['dim_ordering']
    else:
        dim_ordering = K.image_dim_ordering()
    target_size = (data_config['height'], data_config['width'])

    if use_generator:
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGeneratorAcusense(**data_config['train_preprocess'])

        # this is the augmentation configuration we will use for testing:
        test_datagen = ImageDataGeneratorAcusense(**data_config['test_preprocess'])

        np.random.shuffle(samples)

        train_samples = samples[:nb_train_samples]
        val_samples = samples[nb_train_samples:]

        train_generator = train_datagen.flow_from_tuples(
            train_samples, tag_objects,
            target_size=target_size,
            batch_size=data_config['batch_size'],
            class_mode='categorical'
        )

        val_generator = train_datagen.flow_from_tuples(
            val_samples, tag_objects,
            target_size=target_size,
            batch_size=data_config['batch_size'],
            class_mode='categorical'
        )
    else:
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(**data_config['train_preprocess'])

        # this is the augmentation configuration we will use for testing:
        test_datagen = ImageDataGenerator(**data_config['test_preprocess'])

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



