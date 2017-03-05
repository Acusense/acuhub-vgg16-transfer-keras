#### MODEL DEFINITION + LOADING WEIGHTS

template = "vgg16"

if template == "inceptionv3":

    ##### INCEPTION v3

    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)


    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

elif template == "vgg16":

    ###### VGG 16


    from keras.applications import vgg16
    # build the VGG16 network with ImageNet weights
    model = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')

    model.summary()



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





