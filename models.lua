#!/usr/bin/env th


require 'nn'
require 'SpatialConvolutionWithMask'

local M = {}

local conv = nn.SpatialConvolution
local maxpool = nn.SpatialMaxPooling
local avgpool = nn.SpatialAveragePooling
local relu = nn.ReLU
local norm = nn.SpatialCrossMapLRN
local view = nn.View
local linear = nn.Linear
local convPruning = SpatialConvolutionWithMask
local LinearWithMask = nn.LinearWithMask

------------------- Generator functions -------------------
local function _caffe(nclass)
    -- Refer to https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full.prototxt
    model = nn.Sequential()
    model:add(view(-1, 3, 32, 32))
    model:add(conv(3, 32, 5, 5, 1, 1, 2, 2)) -- conv1
    model:add(maxpool(3, 3, 2, 2):ceil()) -- pool1
    model:add(relu(true)) -- relu1
    model:add(norm(3, 5e-05, 0.75)) -- norm1

    model:add(conv(32, 32, 5, 5, 1, 1, 2, 2)) -- conv2
    model:add(relu(true)) -- relu2
    model:add(avgpool(3, 3, 2, 2):ceil()) -- pool2
    model:add(norm(3, 5e-05, 0.75)) -- norm2

    model:add(conv(32, 64, 5, 5, 1, 1, 2, 2)) -- conv3
    model:add(relu(true)) -- relu3
    model:add(avgpool(3, 3, 2, 2):ceil()) -- pool3

    -- ip1
    model:add(view(64*4*4))
    model:add(linear(64*4*4, nclass))

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

local function _allcnnPruning(nclass)
    model = nn.Sequential()
    model:add(view(-1, 3, 32, 32))
    model:add(convPruning(3, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(convPruning(96, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(convPruning(96, 96, 3, 3, 2, 2))
    model:add(relu(true))

    model:add(convPruning(96, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(convPruning(192, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(convPruning(192, 192, 3, 3, 2, 2))
    model:add(relu(true))

    model:add(convPruning(192, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(convPruning(192, 192, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(convPruning(192, 10, 1, 1, 1, 1))
    model:add(relu(true))

    model:add(avgpool(7, 7))
    model:add(view(10*1*1))

    return model
end

local model_generator = {
    caffe = _caffe,
    allcnn = _allcnn,
	allcnnPruning = _allcnnPruning,
}

function M.load(name, nclass)
    return model_generator[name](nclass)
end

return M
