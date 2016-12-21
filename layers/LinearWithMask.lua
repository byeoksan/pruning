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

function LinearWithMask:getAliveWeights()
    local aliveWeights = self.weight[self.weightMask:eq(1)]
    if aliveWeights:dim() == 0 then
        return nil
    end

    return aliveWeights
end

function LinearWithMask:getAliveBiases()
    if not self.bias then
        return nil
    end

    local aliveBiases = self.bias[self.biasMask:eq(1)]
    if aliveBiases:dim() == 0 then
        return nil
    end

    return aliveBiases
end

function LinearWithMask:getAlive()
    local aliveWeights = self:getAliveWeights()
    local aliveBiases = self:getAliveBiases()
    local alive

    if aliveWeights ~= nil and aliveBiases ~= nil then
        alive = aliveWeights:cat(aliveBiases)
    elseif aliveBiases == nil then
        alive = aliveWeights
    elseif aliveWeights == nil then
        alive = aliveBiases
    else
        return nil
    end

    return alive -- 1-D
end

function LinearWithMask:pruneRange(lower, upper)
    if lower > upper then
        lower, upper = upper, lower
    end

    local newWeightMask = torch.cmul(self.weight:ge(lower), self.weight:le(upper))
    self.weightMask[newWeightMask:eq(1)] = 0
    self:applyWeightMask()

    if self.bias then
        local newBiasMask = torch.cmul(self.bias:ge(lower), self.bias:le(upper))
        self.biasMask[newBiasMask:eq(1)] = 0
        self:applyBiasMask()
    end
end

function LinearWithMask:pruneQfactor(qfactor)
    local alive = self:getAlive()
    if alive == nil then
        return
    end

    local mean = alive:mean()
    local std = alive:std()
    local range = torch.abs(qfactor) * std

    lower, upper = mean - range, mean + range
    self:pruneRange(lower, upper)
end

function LinearWithMask:pruneRatio(ratio)
    if ratio > 1 or ratio < 0 then
        return
    end

    local alive = self:getAlive()
    alive = alive:abs():sort()
    local idx = math.floor(alive:size(1) * ratio)

    if idx == 0 then
        return
    end

    local range = alive[idx]
    lower, upper = -range, range
    self:pruneRange(lower, upper)
end

function LinearWithMask:pruneSensitivity(ratio)
    if ratio > 1 or ratio < 0 then
        return
    end

    local ws = self.weightSensitivity
    local bs = self.biasSensitivity

    local sensitivities
    if ws ~= nil and bs ~= nil then
        sensitivities = ws:view(ws:nElement()):cat(bs:view(bs:nElement()))
    elseif ws ~= nil then
        sensitivities = ws:view(ws:nElement())
    elseif bs ~= nil then
        sensitivities = bs:view(bs:nElement())
    else
        return
    end

    sensitivities = sensitivities:sort()
    local idx = math.floor(sensitivities:size(1) * ratio)

    if idx == 0 then
        return
    end

    local upper = sensitivities[idx]

    local newWeightMask = self.weightSensitivity:le(upper)
    self.weightMask[newWeightMask:eq(1)] = 0
    self:applyWeightMask()
    self.weightSensitivity[newWeightMask:eq(1)] = 0

    if self.bias then
        local newBiasMask = self.biasSensitivity:le(upper)
        self.biasMask[newBiasMask:eq(1)] = 0
        self:applyBiasMask()
        self.biasSensitivity[newBiasMask:eq(1)] = 0
    end
end

function LinearWithMask:stash()
    self.stashedWeight = self.weight:clone()
    self.stashedWeightMask = self.weightMask:clone()

    if self.bias then
        self.stashedBias = self.bias:clone()
        self.stashedBiasMask = self.biasMask:clone()
    else
        self.stashedBias = nil
        self.stashedBiasMask = nil
    end
end

function LinearWithMask:stashPop()
    if self.stashedWeight then
        self.weight:copy(self.stashedWeight)
        self.weightMask:copy(self.stashedWeightMask)
    end

    if self.bias and self.stashedBias then
        self.bias:copy(self.stashedBias)
        self.biasMask:copy(self.stashedBiasMask)
    end
end

function LinearWithMask:stashWeightDiff()
    return self.weight - self.stashedWeight
end

function LinearWithMask:stashBiasDiff()
    if self.bias then
        return self.bias - self.stashedBias
    end
end

function LinearWithMask:setSensitivity(weightSensitivity, biasSensitivity)
    self.weightSensitivity = weightSensitivity:clone()
    self.weightSensitivity:abs()
    if biasSensitivity then
        self.biasSensitivity = biasSensitivity:clone()
        self.biasSensitivity:abs()
    else
        self.biasSensitivity = nil
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
