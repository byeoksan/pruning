#!/bin/sh

th main.lua probe-impact \
    -model lenet_2.t7 \
    -data mnist \
    -interval 0.05 \
    -saveName probe_result.t7 \
    -batch 128 \
    -cuda \
    -debug \
    -progress
