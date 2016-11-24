#!/usr/bin/env th

require 'nn'
require 'layers.SpatialConvolutionWithMask'
require 'layers.LinearWithMask'

local M = {}

--local conv = nn.SpatialConvolution
local conv = SpatialConvolutionWithMask
local maxpool = nn.SpatialMaxPooling
local avgpool = nn.SpatialAveragePooling
local relu = nn.ReLU
local norm = nn.SpatialCrossMapLRN
local view = nn.View
--local linear = nn.Linear
local linear = LinearWithMask

------------------- Generator functions -------------------
local function _caffe(nclass)
    -- Refer to https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full.prototxt
    model = nn.Sequential()
    model:add(view(-1, 3, 32, 32))

    conv1 = conv(3, 32, 5, 5, 1, 1, 2, 2)
    conv1:reset(0.0001)
    pool1 = maxpool(3, 3, 2, 2):ceil()
    relu1 = relu(true)
    norm1 = norm(3, 5e-05, 0.75)

    conv2 = conv(32, 32, 5, 5, 1, 1, 2, 2)
    conv2:reset(0.01)
    relu2 = relu(true)
    pool2 = avgpool(3, 3, 2, 2):ceil()
    norm2 = norm(3, 5e-05, 0.75)

    conv3 = conv(32, 64, 5, 5, 1, 1, 2, 2)
    conv3:reset(0.01)
    relu3 = relu(true)
    pool3 = avgpool(3, 3, 2, 2):ceil()

    model:add(conv1)
    model:add(pool1)
    model:add(relu1)
    model:add(norm1)

    model:add(conv2)
    model:add(relu2)
    model:add(pool2)
    model:add(norm2)

    model:add(conv3)
    model:add(relu3)
    model:add(pool3)

    model:add(view(64*4*4))
    ip1 = linear(64*4*4, nclass)
    ip1:reset(0.01)
    model:add(ip1)

    return model
end

local function _allcnn(nclass)
    model = nn.Sequential()
    model:add(view(-1, 3, 32, 32))
    model:add(conv(3, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(conv(96, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(conv(96, 96, 3, 3, 2, 2))
    model:add(relu(true))

    model:add(conv(96, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(conv(192, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(conv(192, 192, 3, 3, 2, 2))
    model:add(relu(true))

    model:add(conv(192, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(conv(192, 192, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(conv(192, 10, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(avgpool(7, 7))
    model:add(view(10*1*1))

    return model
end

local model_generator = {
    caffe = _caffe,
    allcnn = _allcnn,
}

function M.load(name, nclass)
    return model_generator[name](nclass)
end

return M
