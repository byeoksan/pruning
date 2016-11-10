#!/usr/bin/env th

require 'nn'

conv = nn.SpatialConvolution
maxpool = nn.SpatialMaxPooling
relu = nn.ReLU
avgpool = nn.SpatialAveragePooling

model = nn.Sequential()
-- input 3 x 32 x 32 (depth x height x width)
model:add(conv(3, 96, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(96, 96, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(96, 96, 3, 3, 2, 2)):add(relu(true))
model:add(conv(96, 192, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 192, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 192, 3, 3, 2, 2)):add(relu(true))
model:add(conv(192, 192, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 192, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 10, 1, 1, 1, 1)):add(relu(true))
model:add(avgpool(7, 7))
model:add(nn.LogSoftMax())

a = torch.Tensor(3, 32, 32)

print(model:forward(a))
