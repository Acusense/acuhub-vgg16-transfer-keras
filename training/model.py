# Transfer learning: https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# Transfer learning (how it works): https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import vgg16
from data import nb_classes

# build the VGG16 network with ImageNet weights
base_model = vgg16.VGG16(weights='imagenet', include_top=False)
print('Base VGG16 model loaded.')
base_model.summary()


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have n classes as specified by the data
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False



#### LOAD WEIGHTS SELECTIVELY

# load the weights of the VGG16 networks
    # # (trained on ImageNet, won the ILSVRC competition in 2014)
    # # note: when there is a complete match between your model definition
    # # and your weight savefile, you can simply call model.load_weights(filename)
    # assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    # f = h5py.File(weights_path)
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #         # we don't look at the last (fully-connected) layers in the savefile
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()
    # print('Model loaded.')

##########################





