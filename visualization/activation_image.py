# (deprecated) class activation Viz: https://raghakot.github.io/keras-vis/
import os
import numpy as np

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_saliency
from scipy.misc import imsave
from training.model import model
from training.data import target_size
from ..__init__ import config_dict

general_dict = config_dict["general"]

activation_maps_dir = os.path.join(os.environ['BASE_PATH'], 'visualizations', 'activation_maps')
if not os.path.exists(activation_maps_dir):
    os.makedirs(activation_maps_dir)

def activation_vis(layer_name, overlay_image):
    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    # Images corresponding to tiger, penguin, dumbbell, speedboat, spider
    # image_paths = [
    #     "http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
    #     "http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
    #     "https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
    #     "http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
    #     "http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"
    # ]


    # Predict the corresponding class for use in `visualize_saliency`.
    seed_img = utils.load_img(overlay_image, target_size=target_size)
    pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)

    filename = general_dict["model_definition_id"] + "_" + general_dict["model_training_id"] + \
               "_" + layer_name + "_" + overlay_image
    print "Saving activation map for layer %s overlaid onto image %s" % (layer_name, overlay_image)
    imsave(os.path.join(activation_maps_dir, filename), heatmap,'png')
