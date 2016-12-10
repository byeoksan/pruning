-- EE538 Neural Networks
-- Homework 5
-- Convolution Neural Networks 
-- TA. Jihyeon Roh
-- 16.11.02
---------------------------------------------------------------------
require 'torch'
--require 'cunn'
require 'nn'
require 'optim'
require 'lfs'
require 'image'
require 'LinearWithMask'
require 'SpatialConvolutionWithMask'
opt = {}
opt.cuda = false
opt.saveDir = 'model/'
opt.model = 'lenet' -- 'cnnA' 'cnnB' or 'mlp' or 'lenet' for debugging
opt.batchsize = 100 -- mini-batch size (1=pure stochastic)
opt.max_iters = 100
print(opt)
sgd_params = {
   learningRate = 1e-2,
   weightDecay = 1e-3,
}
if opt.saveDir then
	lfs.mkdir(opt.saveDir)
end
torch.manualSeed(1)
-----------------------------------------------------------------------
-- 1. Data load and normalization
-----------------------------------------------------------------------
print('\n1. Data load and normalization ')

mnist  = require('mnist')
traindataset = mnist.traindataset()
testset = mnist.testdataset()

print(traindataset)
print(testset)

-- Zero mean, unit variance normalization
traindataset.data = traindataset.data:double()
mean = traindataset.data:mean()
std = traindataset.data:std()
train_zm = traindataset.data:add(-mean) -- mean subtraction
train_nr = train_zm:div(std) -- std scaling

-- Split traindataset to train/valid data
trainset = {
    size = 50000,
    data = train_nr[{{1,50000}}]:double(), 
    label = traindataset.label[{{1,50000}}]
}

validationset = {
    size = 10000,
    data = train_nr[{{50001,60000}}]:double(),
    label = traindataset.label[{{50001,60000}}]
}

testset.data = testset.data:double()
testset.data:add(-mean)
testset.data:div(std)

-----------------------------------------------------------------------
-- 2. Create the model (Fill in HERE)
-----------------------------------------------------------------------
print('\n2. Create the model ')
if opt.model == 'mlp' then
	model = nn.Sequential()
	model:add(nn.Reshape(28*28))
	model:add(nn.Linear(28*28, 30))
	model:add(nn.Tanh())
	model:add(nn.Linear(30, 10))
	model:add(nn.LogSoftMax())
	print('Network \n'..model:__tostring())
elseif opt.model == 'cnnA' then
	-- Fill in HERE
    model = nn.Sequential()
    model:add(nn.Reshape(1,28,28))
    --conv1 7*7 @6  output: 22*22 @ 6
    model:add(SpatialConvolutionWithMask(1,6,7,7))
    --pool1 2*2     output: 11*11 @ 6
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Reshape(11*11*6))
    --fc1 40 hidden 
    model:add(LinearWithMask(11*11*6, 40))
    model:add(LinearWithMask(40,10))
    model:add(nn.LogSoftMax())

	print('Network \n'..model:__tostring())
elseif opt.model == 'cnnB' then
    model = nn.Sequential()
    model:add(nn.Reshape(1,28,28))
    --conv1 5*5 @6  output: 24*24 @ 6
    model:add(nn.SpatialConvolution(1,6,5,5))
    --pool1 2*2     output: 12*12 @ 6
    model:add(nn.SpatialMaxPooling(2,2))
    --conv2 3*3 @6  output: 10*10 @ 3
    model:add(nn.SpatialConvolution(6,3,3,3))
    --pool1 2*2     output: 5*5 @ 3
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Reshape(5*5*3))
    model:add(nn.Linear(5*5*3, 30))
    model:add(nn.Linear(30,10))
    model:add(nn.LogSoftMax())

    print('Network \n'..model:__tostring())

elseif opt.model == 'lenet' then
    model = nn.Sequential()
    model:add(nn.View(1,28,28))
    --model:add(nn.SpatialConvolution(1,6,5,5)) -- (1x28x28) goes in, (6x24x24) goes out -- Conv1
    model:add(SpatialConvolutionWithMask(1,6,5,5))  -- (1x28x28) goes in, (6x24x24) goes out -- Conv1
    model:add(nn.SpatialMaxPooling(2,2,2,2))    -- (6x24x24) goes out, (6x12x12) goes out
    model:add(nn.ReLU())
    --model:add(nn.SpatialConvolution(6,16,3,3))    -- (6x12x12) goes in, (16x10x10) goes out -- Conv2
    model:add(SpatialConvolutionWithMask(6,16,3,3)) -- (6x12x12) goes in, (16x10x10) goes out -- Conv2
    model:add(nn.SpatialMaxPooling(2,2,2,2))    -- (16x10x10) goes in, (16x5x5) goes out
    model:add(nn.ReLU())
    model:add(nn.View(16*5*5))
    --model:add(nn.Linear(16*5*5, 120))
    model:add(LinearWithMask(16*5*5, 120))
    model:add(nn.ReLU())
    --model:add(nn.Linear(120,84))
    model:add(LinearWithMask(120,84))
    model:add(nn.ReLU())
    --model:add(nn.Linear(84,10))
    model:add(LinearWithMask(84,10))
    model:add(nn.LogSoftMax())
    print('Network \n'..model:__tostring())

end

-- using the negative log likelihood criterion (Fill in here)
criterion =  nn.ClassNLLCriterion()

if opt.cuda then
	-- CPU -> GPU
	model = model:cuda()
	criterion = criterion:cuda()
	trainset.data = trainset.data:cuda()
	trainset.label = trainset.label:cuda()
	validationset.data = validationset.data:cuda()
	validationset.label = validationset.label:cuda()
	testset.data = testset.data:cuda()
	testset.label = testset.label:cuda()		
end


-----------------------------------------------------------------------
-- Trainer
-----------------------------------------------------------------------

x, dl_dx = model:getParameters()
--print (model:findModules('SpatialConvolutionWithMask'))
--testmodel = model:findModules('LinearWithMask')
--print(testmodel)
--testmodel[1]:noBias()
function step(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    
    model:training()
    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size, 28, 28)
        local targets = torch.Tensor(size)
        
        if opt.cuda then
        	inputs = inputs:cuda()
        	targets = targets:cuda()
        end
        
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
        
        -- forward & backward & update
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end
        
        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end

function evaluation(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 200
    
    model:evaluate()
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        
        if opt.cuda then
        	targets = targets:cuda()
        end
        
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

-----------------------------------------------------------------------
-- Train the model (You could modify this part)
-----------------------------------------------------------------------
print('\n3. Train the model ')

local last_accuracy = 0
local decreasing = 0
local threshold = 1 -- how many deacreasing epochs we allow
for i = 1,opt.max_iters do
    local loss = step(opt.batchsize)
--    print(string.format('Epoch %d , Train loss: %4f', i, loss))
    local accuracy = evaluation(validationset,opt.batchsize)
    print(string.format('Epoch %d , Train loss: %4f          Validation accuracy: %.2f', i, loss, accuracy*100))
--    print(string.format('          Validation accuracy: %.2f', accuracy*100))
    
    -- Early stopping
    if accuracy < last_accuracy then
        if decreasing > threshold then break end
        decreasing = decreasing + 1
    else
        decreasing = 0
    end
    last_accuracy = accuracy
end

conv_nodes = model:findModules('nn.SpatialConvolution')
print(conv_nodes)
for i = 1, #conv_nodes do
  print(conv_nodes[i].weight)
end

print('\n4. Evaluate the model ')
local accuracy_test = evaluation(testset)
print(string.format('\nTest accuracy: %.2f \n', accuracy_test*100))



--[[
print '==> visualizing ConvNet filters'
print('Layer 1 filters:')
itorch.image(model:get(1).weight)
print('Layer 2 filters:')
itorch.image(model:get(5).weight)
--]]
-----------------------------------------------------------------------
-- 4. Save the model
-----------------------------------------------------------------------


filename = opt.saveDir..'cnn-mnist-model.t7'
print('5. Save the model in '..filename..'\n')

if opt.cuda then
	model = model:double()
end

torch.save(filename, model)

