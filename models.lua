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

------------------- Generator functions -------------------
local function _cnnA()
    local model = nn.Sequential()
    model:add(view(1, 28, 28))
    model:add(conv(1, 6, 7, 7))
    model:add(maxpool(2, 2))
    model:add(view(11*11*6))
    model:add(linear(11*11*6, 40))
    model:add(linear(40, 10))
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
    return model
end

local model_generator = {
    cnnA = _cnnA,
    cnnB = _cnnB,
    lenet = _lenet,
}

function M.load(name)
    return model_generator[name]()
end

function M.restore(path)
    return torch.load(path)
end

function M.get_prunables(model)
    local prunable_layers = List:new{}
    for i, layer in ipairs(model.modules) do
        if prunables:count(torch.typename(layer)) > 0 then
            prunable_layers:append(layer)
        end
    end

    return prunable_layers
end

return M
