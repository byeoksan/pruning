#!/bin/sh

th main.lua test \
    -model lenet_2.t7
    -data mnist \
    -batch 128 \
    -cuda \
    -progress \
    -debug
