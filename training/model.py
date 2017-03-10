# Transfer learning: https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# Transfer learning (how it works): https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import os

model_vis_dir = os.path.join(os.environ['BASE_PATH'], 'model_vis/')
if not os.path.exists(model_vis_dir):
    os.makedirs(model_vis_dir)

################# MODEL DEFINITION #############################
import sys
sys.path.append(os.environ['BASE_PATH'])
from model_def import model

#######################################################################

# XML rendering (lame): https://github.com/mdaines/viz.js/
# D3 rendering: https://github.com/mstefaniuk/graph-viz-d3-js

######## MODEL VISUALIZATION ################
print "creating model vis png"
from keras.utils.visualize_util import plot
plot(model, to_file=model_vis_dir + 'model.png', show_shapes=True, show_layer_names=True)

print "creating model vis raw graphviz to txt"
from keras.utils.visualize_util import model_to_dot

graphviz_dot = model_to_dot(model)
raw_dot_language = graphviz_dot.to_string()
with open(model_vis_dir + 'model_dot.txt','wb') as f:
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





