#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "No argument supplied specifying the layer to visualize"
else
    THEANO_FLAGS=device=gpu0,floatX=float32 python /acuhub/main.py filter-vis --layer-name $1
fi
