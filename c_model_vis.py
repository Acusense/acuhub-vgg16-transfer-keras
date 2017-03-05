from b_model_def import model

######## MODEL VISUALIZATION ################
print "creating model vis"
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print "show in ipython notebook"
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))