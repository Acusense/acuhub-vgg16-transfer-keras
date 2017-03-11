import os
from training.model_def import model

model_vis_dir = os.path.join(os.environ['BASE_PATH'], 'model_vis/')
if not os.path.exists(model_vis_dir):
    os.makedirs(model_vis_dir)

######## MODEL VISUALIZATION ################
# XML rendering (lame): https://github.com/mdaines/viz.js/
# D3 rendering: https://github.com/mstefaniuk/graph-viz-d3-js

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
