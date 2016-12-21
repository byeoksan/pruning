#!/bin/sh

th main.lua prune \
    -model lenet_2.t7 \
    -init 0.5,0.5,0.5,0.5,0.5 \
    -mult 0.9,0.9,0.9,0.9,0.9 \
    -group individual \
    -method qfactor \
    -batch 128 \
    -saveEpoch 0 \
    -epochs 10 \
    -saveName lenet_pruned \
    -learningRate 0.01 \
    -learningRateDecay 0.0001 \
    -weightDecay 0.0005 \
    -momentum 0.9 \
    -stepIter 1 \
    -data mnist \
    -nosave \
    -debug \
    -progress \
    -cuda
