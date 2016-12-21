#!/bin/sh

th main.lua train \
    -model lenet_2.t7 \
    -modelType lenet \
    -saveName lenet_test \
    -learningRate 0.01 \
    -learningRateDecay 0.0001 \
    -weightDecay 0.0005 \
    -momentum 0.9 \
    -data mnist \
    -nclass 10 \
    -epochs 10 \
    -batch 128 \
    -saveEpoch 0 \
    -cuda \
    -progress \
    -debug \
    -nosave
