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
   self.weight_init = self.weight
  -- print(self.weight_init)
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

function LinearWithMask:prune(qfactor)
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

--debug function
--[[
function count_one(input_weight)
    local input_size = input_weight:size()
    local count = 0
    print (input_size)
    for i = 1, input_size[1] do
        if input_weight[i] == 1 then
            count  = count +1
        end
    end
    print (count)
end
--]]

function LinearWithMask:prune_ratio(qfactor)
   
    local alive_weights = self.weight[self.weightMask:eq(1)]
    if self.bias then
        local alive_bias = self.bias[self.biasMask:eq(1)]
        
        if alive_bias:dim() > 0 then
            alive_weights = alive_weights:cat(alive_bias)
        end
    end
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
