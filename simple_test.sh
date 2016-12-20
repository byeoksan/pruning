#!/bin/bash
th main.lua  onebyone -multiplier 0.5 -qFactor 1,2,3,4,5 -step 4 -model lenet_2.t7 -epochs 2 -saveName lenet_onebyone -cuda -progress
