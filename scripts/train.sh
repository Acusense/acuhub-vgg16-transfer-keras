#!/usr/bin/env bash

THEANO_FLAGS=device=gpu0,floatX=float32 python /acuhub/main.py train
