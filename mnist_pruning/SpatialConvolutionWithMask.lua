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
    self.weight_init = self.weight:clone()
    self.bias_init = self.bias:clone()
    self.weight_square_sum = torch.Tensor(self.weight:size()):zero()
    self.bias_square_sum = torch.Tensor(self.bias:size()):zero()
    
    --print(self.weight)
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

function SpatialConvolutionWithMask:get_alive_weights()
    local alive_weights = self.weight[self.weightMask:eq(1)]
    if alive_weights:dim() == 0 then
        return nil
    end

    return alive_weights
end

function SpatialConvolutionWithMask:get_alive_biases()
    if not self.bias then
        return nil
    end

    local alive_biases = self.bias[self.biasMask:eq(1)]
    if alive_biases:dim() == 0 then
        return nil
    end

    return alive_biases
end

function SpatialConvolutionWithMask:get_alive()
    local alive_weights = self:get_alive_weights()
    local alive_biases = self:get_alive_biases()
    local alive

    if alive_weights ~= nil and alive_biases ~= nil then
        alive = alive_weights:cat(alive_biases)
    elseif alive_biases == nil then
        alive = alive_weights
    elseif alive_weights == nil then
        alive = alive_biases
    else
        return nil
    end

    return alive -- 1-D
end

function SpatialConvolutionWithMask:prune_range(lower, upper)
    if lower > upper then
        lower, upper = upper, lower
    end

    local new_weight_mask = torch.cmul(self.weight:ge(lower), self.weight:le(upper))
    self.weightMask[new_weight_mask:eq(1)] = 0
    self:applyWeightMask()

    if self.bias then
        local new_bias_mask = torch.cmul(self.bias:ge(lower), self.bias:le(upper))
        self.biasMask[new_bias_mask:eq(1)] = 0
        self:applyBiasMask()
    end
end

function SpatialConvolutionWithMask:prune_qfactor(qfactor)
    local alive = self:get_alive()
    if alive == nil then
        return
    end

    local mean = alive:mean()
    local std = alive:std()
    local range = torch.abs(qfactor) * std

    lower, upper = mean - range, mean + range
    self:prune_range(lower, upper)
end

function SpatialConvolutionWithMask:prune_ratio(ratio)
    if ratio > 1 or ratio < 0 then
        return
    end

    local alive = self:get_alive()
    alive = alive:abs():sort()
    local idx = math.floor(alive:size(1) * ratio)

    if idx == 0 then
        return
    end

    local range = alive[idx]
    lower, upper = -range, range
    self:prune_range(lower, upper)
end

function SpatialConvolutionWithMask:clone_weight()
    self.weight_clone = self.weight:clone()
    self.bias_clone = self.bias:clone()
end

function SpatialConvolutionWithMask:weight_diff_square()
    self.weight_diff = self.weight_clone - self.weight
    self.weight_diff:cmul(self.weight_diff); -- weight_diff ^ 2
    self.weight_square_sum = self.weight_square_sum + self.weight_diff

    self.bias_diff = self.bias_clone - self.bias
    self.bias_diff:cmul(self.bias_diff); -- bias_diff ^ 2
    self.bias_square_sum = self.bias_square_sum + self.bias_diff
     
end

function SpatialConvolutionWithMask:cal_sensitivity()
    local tmp_weight = self.weight:clone()
    local tmp_weight_square_sum = self.weight_square_sum:clone()
    local tmp_weight_diff = tmp_weight-self.weight_init

    local tmp_bias = self.bias:clone()
    local tmp_bias_square_sum = self.bias_square_sum:clone()
    local tmp_bias_diff = tmp_bias-self.bias_init

    self.sensitivity_weight = tmp_weight_square_sum:cmul(tmp_weight)
    self.sensitivity_weight:cdiv(tmp_weight_diff)

    self.sensitivity_bias = tmp_bias_square_sum:cmul(tmp_bias)
    self.sensitivity_bias:cdiv(tmp_bias_diff)
    
    local wei_resize_mask = torch.DoubleTensor(self.sensitivity_weight:size()):fill(1)
    local bia_resize_mask = torch.DoubleTensor(self.sensitivity_bias:size()):fill(1)
    
    local resize_sen_wei = self.sensitivity_weight[wei_resize_mask:eq(1)]
    local resize_sen_bia = self.sensitivity_bias[bia_resize_mask:eq(1)]
    
    self.sensitivity = resize_sen_wei:cat(resize_sen_bia)
    --print(self.sensitivity_weight[test:eq(1)])
   -- print(self.sensitivity_weight)
   -- print(self.sensitivity_bias)
   -- self.sensitivity = self.sensitivity_weight:cat(self.sensitivity_bias)

end

function SpatialConvolutionWithMask:prune_sensitivity(ratio)
    if ratio > 1 or ratio < 0 then
        return
    end

    local alive = self.sensitivity
    alive = alive:abs():sort()

    local idx = math.floor(alive:size(1) * ratio)
    if idx == 0 then
        return
    end

    local range = alive[idx]
    lower, upper = -range, range
    self:prune_sense_range(lower, upper)

end

function SpatialConvolutionWithMask:prune_sense_range(lower, upper)
    if lower > upper then
        lower, upper = upper, lower
    end

    local new_weight_mask = torch.cmul(self.sensitivity_weight:ge(lower), self.sensitivity_weight:le(upper))
    self.weightMask[new_weight_mask:eq(1)] = 0
    self:applyWeightMask()

    if self.bias then
        local new_bias_mask = torch.cmul(self.sensitivity_bias:ge(lower), self.sensitivity_bias:le(upper))
        self.biasMask[new_bias_mask:eq(1)] = 0
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
