#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "No argument supplied specifying the image path to overlay the filter vis"
else
    THEANO_FLAGS=device=gpu0,floatX=float32 python /acuhub/main.py filter_vis_image --image-path $1
fi


