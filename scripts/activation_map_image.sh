#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "No argument supplied specifying the layer to visualize"
fi

if [ -z "$2" ]; then
    echo "No argument supplied specifying the image path to overlay the activation map"
else
    THEANO_FLAGS=device=gpu0,floatX=float32 python /acuhub/main.py activation_map_image --layer-name $1 --image-path $2
fi