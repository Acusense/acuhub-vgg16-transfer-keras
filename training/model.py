# Transfer learning: https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# Transfer learning (how it works): https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
import json
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from data import nb_classes
from data import config_path

model_config = json.load(open(config_path))['model']
template = model_config['template']

if template == "inceptionv3":

    from keras.applications.inception_v3 import InceptionV3

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    print('Base Inception v3 model loaded.')

elif template == "vgg16":

    from keras.applications import vgg16
    # build the VGG16 network with ImageNet weights
    base_model = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Base VGG16 model loaded.')
    base_model.summary()

elif template == "vgg19":

    from keras.applications import vgg19
    base_model = vgg19.VGG19(weights='imagenet', include_top=False)
    print('Base VGG19 model loaded.')
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
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


# XML rendering (lame): https://github.com/mdaines/viz.js/
# D3 rendering: https://github.com/mstefaniuk/graph-viz-d3-js

######## MODEL VISUALIZATION ################
print "creating model vis png"
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print "creating model vis raw graphviz to txt"
from keras.utils.visualize_util import model_to_dot

graphviz_dot = model_to_dot(model)
raw_dot_language = graphviz_dot.to_string()
with open('model_dot.txt','wb') as f:
    f.write(raw_dot_language)

# from IPython.display import SVG
# SVG(graphviz_dot.create(prog='dot', format='svg'))

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





