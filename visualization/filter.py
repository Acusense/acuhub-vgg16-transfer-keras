# (working) Filter viz: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
# How does it work: https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59#.asjppsubd
# https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb
# How it works w/ keras: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import os, time
from training.model import model
from keras import backend as K
from __init__ import general_dict, visualizer_dict

filter_vis_dict = visualizer_dict["filter"]

filter_vis_dir = os.path.join(os.environ['BASE_PATH'], 'visualizations', 'filter_vis')
if not os.path.exists(filter_vis_dir):
    os.makedirs(filter_vis_dir)

'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes (with the TensorFlow backend).
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
def filter_vis(layer_name, filter_index=None):

    # util function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    print('layers: %s' % str(layer_dict.keys()))

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


    kept_filters = []

    layer_output = layer_dict[layer_name].output
    nb_filters = layer_dict[layer_name].nb_filter

    print('number of filters in layer %s: %d' % (layer_name, nb_filters))
    if filter_index == None:
        filter_indices = range(0, nb_filters)
    else:
        filter_indices = [filter_index]

    for ind in filter_indices:
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % ind)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered

        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, ind, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, ind])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            input_img_data = np.random.random((1, 3, filter_vis_dict['width'], filter_vis_dict['height']))
        else:
            input_img_data = np.random.random((1, filter_vis_dict['width'], filter_vis_dict['height'], 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (ind, end_time - start_time))

        print('Saving image from layer %s and filter index %s' % (layer_name, ind))
        filename = general_dict['model_description_id'] + '_' +\
                   general_dict['model_training_id'] + '_' + \
                   'layer_%s_filter_%d.png' % (layer_name, ind)
        imsave(os.path.join(filter_vis_dir, filename), img, 'png')


    # # we will stich the best 64 filters on a 8 x 8 grid.
    # n = 8
    #
    # # the filters that have the highest loss are assumed to be better-looking.
    # # we will only keep the top 64 filters.
    # kept_filters.sort(key=lambda x: x[1], reverse=True)
    # kept_filters = kept_filters[:n * n]
    #
    # # build a black picture with enough space for
    # # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    # margin = 5
    # width = n * filter_vis_dict['width'] + (n - 1) * margin
    # height = n * filter_vis_dict['height'] + (n - 1) * margin
    # stitched_filters = np.zeros((width, height, 3))
    #
    # # fill the picture with our saved filters
    # for i in range(n):
    #     for j in range(n):
    #         img, loss = kept_filters[i * n + j]
    #         stitched_filters[(filter_vis_dict['width'] + margin) * i: (filter_vis_dict['width'] + margin) * i + filter_vis_dict['width'],
    #                          (filter_vis_dict['height'] + margin) * j: (filter_vis_dict['height'] + margin) * j + filter_vis_dict['height'], :] = img
    #
    # # save the result to disk
    # imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)