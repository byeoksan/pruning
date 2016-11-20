#!/usr/bin/env th

--[[
	Supervised Learning Recipe
	1. Preprocess data to facilitate learning
	2. Describe a model to solve a classification task
	3. Choose a loss function to minimize
	4. Define a sampling procedure(stochastic, mini-batches),
	   and apply one of several optimization techniques to train the model's parameters.
	5. Estimate the model's performance on unseen data(test data).
]]--

require 'torch'
require 'nn'
require 'optim'
require 'lfs'
matio = require 'matio'
optnet = require 'optnet'
datasrc = require 'datasrc.lua'

optParam = {
	batchsize = 1000,
	maxIter = 50,
	cuda = true
}
sgdParam = {
	learningRate = 1e-1,
	weightDecay = 1e-3
}
print('\n------------------------------------------------')
print('* Learning Setup')
print(optParam, sgdParam, '------------------------------------------------\n')

torch.manualSeed(1)
--------------------------------------------------------------------------------
-- 1. Data load and Normalization
--------------------------------------------------------------------------------
trainData, testData = datasrc.init()	-- data load
sizeParam = {
	sizetr = 40000,
	sizevalid = 10000,
	sizetest = 10000
}
trainset = {
	size = sizeParam.sizetr,
	data = trainData.data[{ {1,sizeParam.sizetr} }],
	labels = trainData.labels[{ {1,sizeParam.sizetr} }]
}
validationset = {
	size = sizeParam.sizevalid,
	data = trainData.data[{ {1,sizeParam.sizevalid} }],
	labels = trainData.labels[{ {1,sizeParam.sizevalid} }]
}
testset = {
	size = sizeParam.sizetest,
	data = testData.data[{ {1,sizeParam.sizetest} }],
	labels = testData.labels[{ {1,sizeParam.sizetest} }]
}

--------------------------------------------------------------------------------
-- 2. Model Define
--------------------------------------------------------------------------------
conv = nn.SpatialConvolution
maxpool = nn.SpatialMaxPooling
relu = nn.ReLU
avgpool = nn.SpatialAveragePooling

-- input 3 x 32 x 32 (depth x height x width)
model = nn.Sequential()
model:add(conv(3, 96, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(96, 96, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(96, 96, 3, 3, 2, 2)):add(relu(true))
model:add(conv(96, 192, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 192, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 192, 3, 3, 2, 2)):add(relu(true))
model:add(conv(192, 192, 3, 3, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 192, 1, 1, 1, 1)):add(relu(true))
model:add(conv(192, 10, 1, 1, 1, 1)):add(relu(true))
model:add(avgpool(7, 7))	-- tensor dimension after avgpl(7,7): 10x1x1
model:add(nn.View(10*1*1))
--model:add(nn.LogSoftMax())
model:add(nn.SoftMax())  -- prob
criterion = nn.ClassNLLCriterion()

if optParam.cuda then
	require 'cunn'
	model = model:cuda()
	criterion = nn.ClassNLLCriterion():cuda()
	trainset.data = trainset.data:cuda()
	trainset.labels = trainset.labels:cuda()
	validationset.data = validationset.data:cuda()
	validationset.labels = validationset.labels:cuda()
	testset.data = testset.data:cuda()
	testset.labels = testset.labels:cuda()
end


print('\n------------------------------------------------')
print('* Network\n'..model:__tostring())
print('* Memory used before training')
print(optnet.countUsedMemory(model))
print('------------------------------------------------\n')


--------------------------------------------------------------------------------
-- 3.1 Trainer Define
--------------------------------------------------------------------------------
x, dl_dx = model:getParameters()

function step(batch_size)	-- return loss normalized by # of batches
	local loss_cur = 0
	local cnt = 0
	local shuffle = torch.randperm(trainset.size)
	batch_size = batch_size or 500

	for t = 1, trainset.size, batch_size do
		local size = math.min(t + batch_size - 1, trainset.size) - t
		local inputs = torch.Tensor(size, 3, 32, 32)
		local targets = torch.Tensor(size)

		if optParam.cuda then
			inputs = inputs:cuda()
			targets = targets:cuda()
		end

		for i = 1, size do
			local input = trainset.data[shuffle[i+t]]
			local target = trainset.labels[shuffle[i+t]]
			inputs[i] = input
			targets[i] = target
		end

		local feval = function(x_new)
			if x ~= x_new then x:copy(x_new) end
			dl_dx:zero()

			local loss = criterion:forward(model:forward(inputs), targets)
			model:backward(inputs, criterion:backward(model.output, targets))

			return loss, dl_dx
		end

		_, fs = optim.sgd(feval, x, sgdParam)
		cnt = cnt + 1
		loss_cur = loss_cur + fs[1]
	end

	return loss_cur / cnt
end

function evaluation(dataset, batch_size)
	local cnt_correct = 0
	batch_size = batch_size or 500

	for i = 1, dataset.size, batch_size do
		local size = math.min(i + batch_size - 1, dataset.size) - i
		local inputs = dataset.data[{ {i,i+size-1} }]
		local targets = dataset.labels[{ {i,i+size-1} }]

		local outputs = model:forward(inputs)
		local _, indices = torch.max(outputs, 2)
		local cnt_right = indices:eq(targets:long()):sum()
	
		cnt_correct = cnt_correct + cnt_right
	end

	return cnt_correct / dataset.size
end


--------------------------------------------------------------------------------
-- 3.2 Train Model
--------------------------------------------------------------------------------
print('\n------------------------------------------------')
print('* Training')

local loss_table_train = {}
local acc_table_train = {}
local acc_table_valid = {}

--[[
local ealrystopParam = {
	last_acc_valid = 0,
	decreasing = 0,
	threshold = 2
}
]]--

last_acc_valid = 0
decreasing = 0
threshold = 2

for i = 1, optParam.maxIter do
	local loss = step(optParam.batchsize)
	print(string.format('Epoch %d,', i))
	print(string.format('	Train loss		: %.8f', loss))
	local acc_train = evaluation(trainset, optParam.batchsize)
	print(string.format('	Train Accuracy		: %.8f', acc_train*100))
	local acc_valid = evaluation(validationset, optParam.batchsize)
	print(string.format('	Validation Accuracy	: %.8f', acc_valid*100))
	
	table.insert(loss_table_train, loss)
	table.insert(acc_table_train, acc_train*100)
	table.insert(acc_table_valid, acc_valid*100)
	
	-- Early Stop
	if acc_valid < last_acc_valid then
		if decreasing > threshold then break end
		decreasing = decreasing + 1
	else
		decreasing = 0
	end
	last_acc_valid = acc_valid
end

print('* Memory Used After Training')
print(optnet.countUsedMemory(model))
print('------------------------------------------------\n')

print('\n------------------------------------------------')
print('* Evaluation')
local acc_test = evaluation(testset)
print(string.format('	Test Accuracy		: %.8f', acc_test*100))
print('------------------------------------------------\n')


