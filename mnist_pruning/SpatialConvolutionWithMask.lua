local THNN = require 'nn.THNN'
local SpatialConvolutionWithMask, parent = torch.class('SpatialConvolutionWithMask', 'nn.SpatialConvolution')

function SpatialConvolutionWithMask:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

    if not self.weightMask then
        self.weightMask = torch.DoubleTensor(self.weight:size()):fill(1)
    end

    if self.bias and not self.biasMask then
        self.biasMask = torch.DoubleTensor(self.bias:size()):fill(1)
    end
end

function SpatialConvolutionWithMask:noBias()
    parent.noBias(self)
    self.biasMask = nil
    return self
end

function SpatialConvolutionWithMask:reset(stdv)
    parent.reset(self, stdv)
    if not self.weightMask then
        self.weightMask = torch.DoubleTensor(self.weight:size()):fill(1)
    else
        self.weightMask:fill(1)
    end
    if self.bias then
        if self.biasMask then
            self.biasMask:fill(1)
        else
            self.biasMask = torch.DoubleTensor(self.bias:size()):fill(1)
        end
    end
end

function SpatialConvolutionWithMask:weightMaskSet(threshold)
    self.weightMask:cmul(self.weight:ge(threshold):double())
    self:applyWeightMask()
end

function SpatialConvolutionWithMask:biasMaskSet(threshold)
    if self.bias then
        self.biasMask:cmul(self.bias:ge(threshold):double())
        self:applyBiasMask()
    end
end

function SpatialConvolutionWithMask:applyWeightMask()
    self.weight:cmul(self.weightMask)
end

function SpatialConvolutionWithMask:applyBiasMask()
    if self.bias then
        self.bias:cmul(self.biasMask)
    end
end

function SpatialConvolutionWithMask:prune(qfactor)
    local alive_weights = self.weight[self.weightMask:eq(1)]
    if self.bias then
        local alive_bias = self.bias[self.biasMask:eq(1)]
        if alive_bias:dim() > 0 then
            alive_weights = alive_weights:cat(alive_bias)
        end
    end

    local mean = alive_weights:mean()
    local std = alive_weights:std()
    local threshold = torch.abs(qfactor) * std

    local new_weight_mask = torch.abs(self.weight - mean):ge(threshold)
    self.weightMask[new_weight_mask:eq(0)] = 0
    self:applyWeightMask()

    if self.bias then
        local new_bias_mask = torch.abs(self.bias - mean):ge(threshold)
        self.biasMask[new_bias_mask:eq(0)] = 0
        self:applyBiasMask()
    end
end

function find_threshold(input_weight, qfactor)
    total_size = 1
    weight_size = input_weight:size()
    for i = 1, weight_size:size() do
        total_size = total_size * weight_size[i] 
    end
    weight_abs = torch.abs(input_weight)
    resize_weight_abs = weight_abs:view(total_size)
    sort_weight, i = torch.sort(resize_weight_abs)
    prune_index = torch.ceil(qfactor/100 * total_size)
    
    local threshold = sort_weight[prune_index]
    return threshold
end

function SpatialConvolutionWithMask:prune_ratio(qfactor)
    local alive_weights = self.weight[self.weightMask:eq(1)]
    if self.bias then
        local alive_bias = self.bias[self.biasMask:eq(1)]
        if alive_bias:dim() > 0 then
            alive_weights = alive_weights:cat(alive_bias)
        end
    end
--    local mean = alive_weights:mean()
 --   local std = alive_weights:std()
 --   local threshold = torch.abs(qfactor) * std

    local threshold_weight = find_threshold(self.weight, qfactor)
    local threshold_bias = find_threshold(self.bias, qfactor)

    local new_weight_mask = torch.abs(self.weight):gt(threshold_weight)
    self.weightMask[new_weight_mask:eq(0)] = 0
    self:applyWeightMask()

    if self.bias then
        local new_bias_mask = torch.abs(self.bias):gt(threshold_bias)
        self.biasMask[new_bias_mask:eq(0)] = 0
        self:applyBiasMask()
    end
end

function SpatialConvolutionWithMask:updateOutput(input)
    return parent.updateOutput(self, input)
end

function SpatialConvolutionWithMask:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self, input, gradOutput)
end

function SpatialConvolutionWithMask:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput, scale)
    self.gradWeight:cmul(self.weightMask)
end
