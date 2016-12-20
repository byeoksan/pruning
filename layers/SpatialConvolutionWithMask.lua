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

function SpatialConvolutionWithMask:getAliveWeights()
    local aliveWeights = self.weight[self.weightMask:eq(1)]
    if aliveWeights:dim() == 0 then
        return nil
    end

    return aliveWeights
end

function SpatialConvolutionWithMask:getAliveBiases()
    if not self.bias then
        return nil
    end

    local aliveBiases = self.bias[self.biasMask:eq(1)]
    if aliveBiases:dim() == 0 then
        return nil
    end

    return aliveBiases
end

function SpatialConvolutionWithMask:getAlive()
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

function SpatialConvolutionWithMask:pruneRange(lower, upper)
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

function SpatialConvolutionWithMask:pruneQfactor(qfactor)
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

function SpatialConvolutionWithMask:pruneRatio(ratio)
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

function SpatialConvolutionWithMask:stash()
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

function SpatialConvolutionWithMask:stashPop()
    if self.stashedWeight then
        self.weight:copy(self.stashedWeight)
        self.weightMask:copy(self.stashedWeightMask)
    end

    if self.bias and self.stashedBias then
        self.bias:copy(self.stashedBias)
        self.biasMask:copy(self.stashedBiasMask)
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
