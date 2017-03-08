# Training Visualization Acusense (working): https://github.com/acusense/acuhub-viz

# Original code for ref (non-working) : https://github.com/fchollet/hualos
import json
from model import template, model
from data import nb_train_samples, nb_val_samples
from data import train_generator, val_generator
from data import config_path

training_config = json.load(open(config_path))['training']

def train():
    if template == "inceptionv3":
        from keras import callbacks

        # compile the model with all of the training parameters (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss=training_config['loss_function'],
                  metrics=training_config['metrics'])


        # create remote monitor to track training metrics
        remote = callbacks.RemoteMonitor(root='http://localhost:9000')


        # train the model on the new data for a few epochs
        print "training model with full inception model"
        model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=training_config['nb_epoch'],
            validation_data=val_generator,
            nb_val_samples=nb_val_samples,
            callbacks=[remote]
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
    elif template == "vgg16":
        pass

    elif template == "vgg19":
        pass


