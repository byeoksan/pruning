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
