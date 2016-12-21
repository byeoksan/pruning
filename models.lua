#!/usr/bin/env th

require 'nn'

local M = {}

local prunables = List{
    'SpatialConvolutionWithMask',
    'LinearWithMask',
}

for layer in prunables:iterate() do
    require('layers.' .. layer)
end

local conv = SpatialConvolutionWithMask
local maxpool = nn.SpatialMaxPooling
local avgpool = nn.SpatialAveragePooling
local relu = nn.ReLU
local norm = nn.SpatialCrossMapLRN
local view = nn.View
local linear = LinearWithMask
local dropout = nn.Dropout

------------------- MNIST Generator functions -------------------
local function _cnnA()
    local model = nn.Sequential()
    model:add(view(1, 28, 28))
    model:add(conv(1, 6, 7, 7))
    model:add(maxpool(2, 2))
    model:add(view(11*11*6))
    model:add(linear(11*11*6, 40))
    model:add(linear(40, 10))
    model:add(nn.LogSoftMax())
    return model
end

local function _cnnB()
    local model = nn.Sequential()
    model:add(view(1, 28, 28))
    model:add(conv(1, 6, 5, 5))
    model:add(maxpool(2, 2))
    model:add(conv(6, 3, 3, 3))
    model:add(maxpool(2, 2))
    model:add(view(5*5*3))
    model:add(linear(5*5*3, 30))
    model:add(linear(30, 10))
    model:add(nn.LogSoftMax())
    return model
end

local function _lenet()
    local model = nn.Sequential()
    model:add(view(1, 28, 28))
    model:add(conv(1, 6, 5, 5))
    model:add(maxpool(2, 2, 2, 2))
    model:add(relu(true))
    model:add(conv(6, 16, 3, 3))
    model:add(maxpool(2, 2, 2, 2))
    model:add(relu(true))
    model:add(view(16*5*5))
    model:add(linear(16*5*5, 120))
    model:add(relu(true))
    model:add(linear(120, 84))
    model:add(relu(true))
    model:add(linear(84, 10))
    model:add(nn.LogSoftMax())
    return model
end


------------------- CIFAR Generator functions -------------------
local function _caffe(nclass)
    -- Refer to https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full.prototxt
    model = nn.Sequential()
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
    model:add(nn.LogSoftMax())

    return model
end

local function _allcnn(nclass)
    model = nn.Sequential()
    model:add(conv(3, 96, 3, 3, 1, 1, 1, 1)) -- 96 x 32 x 32
    model:add(relu(true))

    model:add(conv(96, 96, 3, 3, 1, 1, 1, 1)) -- 96 x 32 x 32
    model:add(relu(true))

    --model:add(conv(96, 96, 3, 3, 2, 2, 1, 1)) -- 96 x 15 x 15
    --model:add(relu(true))
    model:add(maxpool(3, 3, 2, 2):floor())

    model:add(conv(96, 192, 3, 3, 1, 1, 1, 1)) -- 192 x 15 x 15
    model:add(relu(true))

    model:add(conv(192, 192, 3, 3, 1, 1, 1, 1)) -- 192 x 15 x 15
    model:add(relu(true))

    --model:add(conv(192, 192, 3, 3, 2, 2, 1, 1)) -- 192 x 6 x 6
    --model:add(relu(true))
    model:add(maxpool(3, 3, 2, 2):ceil())

    model:add(conv(192, 192, 3, 3, 1, 1, 1, 1)) -- 192 x 6 x 6
    model:add(relu(true))

    model:add(conv(192, 192, 1, 1, 1, 1)) -- 192 x 6 x 6
    model:add(relu(true))

    model:add(conv(192, 10, 1, 1, 1, 1)) -- 10 x 6 x 6
    model:add(relu(true))

    model:add(avgpool(6, 6))
    model:add(view(10*1*1))
    model:add(nn.LogSoftMax())

    return model
end

local function _vgg(nclass)
    -- https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua
    local model = nn.Sequential()

    model:add(conv(3, 64, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.3))

    model:add(conv(64, 64, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(maxpool(2, 2, 2, 2):ceil())

    model:add(conv(64, 128, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(128, 128, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(maxpool(2, 2, 2, 2):ceil())

    model:add(conv(128, 256, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(maxpool(2, 2, 2, 2):ceil())

    model:add(conv(256, 512, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(maxpool(2, 2, 2, 2):ceil())

    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.4))

    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(maxpool(2, 2, 2, 2):ceil())

    model:add(view(512))
    model:add(dropout(0.5))
    model:add(linear(512, 512))
    model:add(relu(true))
    model:add(dropout(0.5))
    model:add(linear(512, nclass))
    model:add(nn.LogSoftMax())

    return model
end

local function _myvgg(nclass)
    -- https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    model = nn.Sequential()
    model:add(conv(3, 64, 3, 3, 1, 1, 1, 1)) -- conv1_1
    model:add(relu(true))
    model:add(dropout(0.5)) -- drop1
    model:add(maxpool(2, 2, 2, 2)) -- pool1

    model:add(conv(64, 128, 3, 3, 1, 1, 1, 1)) -- conv2_1
    model:add(relu(true))
    model:add(dropout(0.5)) -- drop2
    model:add(maxpool(2, 2, 2, 2)) -- pool2

    model:add(conv(128, 256, 3, 3, 1, 1, 1, 1)) -- conv3_1
    model:add(relu(true))
    model:add(conv(256, 256, 3, 3, 1, 1, 1, 1)) -- conv3_2
    model:add(relu(true))
    model:add(dropout(0.5)) -- drop3
    model:add(maxpool(2, 2, 2, 2)) -- pool3

    model:add(conv(256, 512, 3, 3, 1, 1, 1, 1)) -- conv4_1
    model:add(relu(true))
    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1)) -- conv4_2
    model:add(relu(true))
    --model:add(dropout(0.5)) -- drop4
    model:add(maxpool(2, 2, 2, 2)) -- pool4

    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1)) -- conv5_1
    model:add(relu(true))
    model:add(conv(512, 512, 3, 3, 1, 1, 1, 1)) -- conv5_2
    model:add(relu(true))
    --model:add(dropout(0.5)) -- drop5
    model:add(maxpool(2, 2, 2, 2)) -- pool5

    model:add(view(512))
    model:add(linear(512, 4096)) -- fc6
    model:add(relu(true))
    model:add(dropout(0.5)) -- drop6
    model:add(linear(4096, 4096)) -- fc7
    model:add(relu(true))
    model:add(dropout(0.5)) -- drop7
    model:add(linear(4096, nclass)) -- fc8
    model:add(nn.LogSoftMax())

    return model
end

local function _exp(nclass)
    model = nn.Sequential()
    model:add(conv(3, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(conv(96, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(conv(96, 96, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.5))
    model:add(maxpool(3, 3, 2, 2))

    model:add(conv(96, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(conv(192, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(conv(192, 192, 3, 3, 1, 1, 1, 1))
    model:add(relu(true))
    model:add(dropout(0.5))
    model:add(maxpool(3, 3, 2, 2))

    model:add(view(192*7*7))
    model:add(linear(192*7*7, 2048))
    model:add(relu(true))
    model:add(dropout(0.5))
    model:add(linear(2048, 2048))
    model:add(relu(true))
    model:add(dropout(0.5))
    model:add(linear(2048, nclass))
    model:add(nn.LogSoftMax())

    return model

end

local model_generator = {
	-- mnist models
    cnnA = _cnnA,
    cnnB = _cnnB,
    lenet = _lenet,

	--cifar models
	caffe = _caffe,
    allcnn = _allcnn,
    vgg = _vgg,
    myvgg = _myvgg,
    exp = _exp,
}

function M.load(name, nclass)
    return model_generator[name](nclass)
end

function M.restore(path)
    return torch.load(path)
end

function M.get_prunables(model)
    local prunable_layers = List:new{}
    for i, layer in ipairs(model.modules) do
        if prunables:contains(torch.typename(layer)) then
            prunable_layers:append(layer)
        end
    end

    return prunable_layers
end

function M.get_prunables_by_type(model)
    local prunable_map = OrderedMap()

    cnt = 0
    for i, layer in ipairs(model.modules) do
        local typename = torch.typename(layer)
        if prunables:contains(typename) then
            cnt = cnt + 1
            if not prunable_map:keys():contains(typename) then
                prunable_map[typename] = List{}
            end
            prunable_map[typename]:append(List{layer, cnt})
        end
    end

    return prunable_map
end

return M
