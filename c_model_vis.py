# XML rendering (lame): https://github.com/mdaines/viz.js/
# D3 rendering: https://github.com/mstefaniuk/graph-viz-d3-js

from b_model_def import model

######## MODEL VISUALIZATION ################
print "creating model vis"
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print "show in ipython notebook"
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

graphviz_dot = model_to_dot(model)
raw_dot_language = graphviz_dot.to_string()
SVG(graphviz_dot.create(prog='dot', format='svg'))
