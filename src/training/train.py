# Training Visualization Acusense (working): https://github.com/acusense/acuhub-viz

# Original code for ref (non-working) : https://github.com/fchollet/hualos
import os, json
from model import model
from data import nb_train_samples, nb_val_samples
from data import train_generator, val_generator
from __init__ import config_path
from keras import callbacks
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
K.get_session().run(tf.global_variables_initializer())

training_config = json.load(open(config_path))['training']
snapshots_dir = os.path.join(os.environ['BASE_PATH'],'snapshots/')
training_filepath = os.path.join(os.environ['BASE_PATH'], 'training_file.csv')

with tf.device(os.environ['TENSORFLOW_DEVICE']):
    def train():

        # compile the model with all of the training parameters (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=training_config['optimizer'], loss=training_config['loss_function'],
                  metrics=training_config['metrics'])


        # create csv logger to store to CSV
        csv_logging = callbacks.CSVLogger(training_filepath, separator=',', append=False)
        model_checkpoint = callbacks.ModelCheckpoint(snapshots_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                     monitor='val_loss', verbose=0,
                                                     save_best_only=False, save_weights_only=False,
                                                     mode='auto', period=1)

        # train the model on the new data for a few epochs
        print "training model with full model"
        model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=training_config['nb_epoch'],
            validation_data=val_generator,
            nb_val_samples=nb_val_samples,
            callbacks=[csv_logging, model_checkpoint]
        )

        # # at this point, the top layers are well trained and we can start fine-tuning
        # # convolutional layers from inception V3. We will freeze the bottom N layers
        # # and train the remaining top layers.

        # # let's visualize layer names and layer indices to see how many layers
        # # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        #    print(i, layer.name)

        # # we chose to train the top 2 inception blocks, i.e. we will freeze
        # # the first 172 layers and unfreeze the rest:
        # for layer in model.layers[:172]:
        #    layer.trainable = False
        # for layer in model.layers[172:]:
        #    layer.trainable = True

        # # we need to recompile the model for these modifications to take effect
        # # we use SGD with a low learning rate
        # from keras.optimizers import SGD
        # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # # we train our model again (this time fine-tuning the top 2 inception blocks
        # # alongside the top Dense layers
        # model.fit_generator(...)



