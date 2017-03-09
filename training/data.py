import os, math
import numpy as np
import json
import keras as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array, array_to_img

class ImageDataGeneratorAcusense(ImageDataGenerator):
    def __init__(self):
        super(ImageDataGeneratorAcusense, self).__init__()


    def flow_from_tuples(self, tuples,
                        target_size=(256, 256), color_mode='rgb',
                        classes=None, class_mode='categorical',
                        batch_size=32, shuffle=True, seed=None,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='jpeg',
                        follow_links=False):
        return TuplesIterator(
            tuples, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)


class TuplesIterator(Iterator):

    def __init__(self, tuples, tag_objects, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.tuples = tuples
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        self.class_indices = {tag_object.name:ind for ind, tag_object in enumerate(tag_objects)}

        self.nb_class = len(self.class_indices.keys())

        for ind, tuple in enumerate(tuples):
            fname = tuple[0]
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filepaths = []
        self.classes = []
        i = 0
        for ind, tuple in enumerate(tuples):
            fname = tuple[0]
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self.classes.append(tuple[1])
                i += 1
                # add filename relative to directory
                self.filepaths.append(os.path.join(data_path, tuple[0]))
        super(TuplesIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            filepath = self.filepaths[j]
            img = load_img(filepath,
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = np.array(self.classes[index_array])
        elif self.class_mode == 'binary':
            batch_y = np.array(self.classes[index_array], dtype=K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype=K.floatx())
            for i, class_list in enumerate(self.classes[index_array]):
                for class_ind in class_list:
                    batch_y[i, class_ind] = 1.
        else:
            return batch_x
        return batch_x, batch_y


data_path = os.path.join(os.environ['BASE_PATH'], 'data')
config_path = os.path.join(os.environ['BASE_PATH'], 'config.json')

data_config = json.load(open(config_path))['data']

samples = data_config['file_tags_map']
tags = data_config['tags']
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

# if dim_ordering == "tf":
#     X_train = np.zeros((nb_train_samples, data_config['width'],
#                        data_config['height'], 3))
#     X_val = np.zeros((nb_val_samples, data_config['width'],
#                      data_config['height'], 3))
# else:
#     X_train = np.zeros((nb_train_samples, 3, data_config['width'],
#                        data_config['height']))
#
#     X_val = np.zeros((nb_val_samples, 3, data_config['width'],
#                      data_config['height']))
# y_train = []
# y_val = []
#
#
# for ind, sample in enumerate(samples):
#     img_path = os.path.join(data_path, sample[0])
#     if ind in range(0, nb_train_samples):
#         X_train[ind] = img_to_array(load_img(img_path, target_size=target_size))
#         y_train.append(sample[1])
#     elif ind in range(nb_train_samples, len(samples)):
#         X_val[ind - nb_train_samples] = img_to_array(load_img(img_path, target_size=target_size))
#         y_val.append(sample[1])
#
# # Convert y to categorical variables
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_val = np_utils.to_categorical(y_val, nb_classes)

# train_generator = train_datagen.flow(
#     X_train,
#     Y_train,
#     batch_size=data_config['batch_size']
# )
#
# val_generator = test_datagen.flow(
#     X_val,
#     Y_val,
#     batch_size=data_config['batch_size']
# )

train_generator = train_datagen.flow_from_tuples(
    train_samples,
    target_size=target_size,
    batch_size=data_config['batch_size'],
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_tuples(
    val_samples,
    target_size=target_size,
    batch_size=data_config['batch_size'],
    class_mode='categorical'
)


