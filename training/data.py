import os, math
import numpy as np
from keras import backend as K
from ..keras_util import ImageDataGeneratorAcusense
from ..__init__ import config_dict

data_path = os.path.join(os.environ['BASE_PATH'], 'data')

data_config = config_dict['data']

samples = data_config['file_tags_map']
tag_objects = data_config['tags']
nb_classes = max(max([sample[1] for sample in samples])) + 1
nb_samples = len(samples)
nb_train_samples = int(math.floor(nb_samples * data_config['train_val_split']))
nb_val_samples = nb_samples - nb_train_samples

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGeneratorAcusense(**data_config['train_preprocess'])

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGeneratorAcusense(**data_config['test_preprocess'])

# craft data and labels from files
if "dim_ordering" in data_config['train_preprocess'].keys():
    dim_ordering = data_config['train_preprocess']['dim_ordering']
else:
    dim_ordering = K.image_dim_ordering()

target_size = (data_config['height'], data_config['width'])

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


