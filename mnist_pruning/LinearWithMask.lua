local THNN = require 'nn.THNN'
local LinearWithMask, parent = torch.class('LinearWithMask', 'nn.Linear')

function LinearWithMask:__init(inputSize, outputSize, bias)
   parent.__init(self, inputSize, outputSize, bias)

   if not self.weightMask then
       self.weightMask = torch.DoubleTensor(self.weight:size()):fill(1)
   end

   if self.bias and not self.biasMask then
       self.biasMask = torch.DoubleTensor(self.bias:size()):fill(1)
   end
end

function LinearWithMask:noBias()
    parent.noBias(self)
    self.biasMask = nil
   return self
end

function LinearWithMask:reset(stdv)
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

function LinearWithMask:weightMaskSet(threshold)
    self.weightMask:cmul(self.weight:ge(threshold):double())
    self:applyWeightMask()
end

function LinearWithMask:biasMaskSet(threshold)
    if self.bias then
        self.biasMask:cmul(self.bias:ge(threshold):double())
        self:applyBiasMask()
    end
end

function LinearWithMask:applyWeightMask()
    self.weight:cmul(self.weightMask)
end

function LinearWithMask:applyBiasMask()
    if self.bias then
        self.bias:cmul(self.biasMask)
    end
end

function LinearWithMask:get_alive_weights()
    local alive_weights = self.weight[self.weightMask:eq(1)]
    if alive_weights:dim() == 0 then
        return nil
    end

    return alive_weights
end

function LinearWithMask:get_alive_biases()
    if not self.bias then
        return nil
    end

    local alive_biases = self.bias[self.biasMask:eq(1)]
    if alive_biases:dim() == 0 then
        return nil
    end

    return alive_biases
end

function LinearWithMask:get_alive()
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

function LinearWithMask:prune_range(lower, upper)
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

function LinearWithMask:prune_qfactor(qfactor)
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

function LinearWithMask:prune_ratio(ratio)
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

function LinearWithMask:updateOutput(input)
    return parent.updateOutput(self, input)
end

function LinearWithMask:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self, input, gradOutput)
end

function LinearWithMask:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput, scale)
    self.gradWeight:cmul(self.weightMask)
end

-- we do not need to accumulate parameters when sharing
LinearWithMask.sharedAccUpdateGradParameters = LinearWithMask.accUpdateGradParameters
