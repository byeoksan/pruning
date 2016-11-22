#!/usr/bin/env th


require 'torch'
require 'cutorch'
require 'nn'
require 'optim'

models = require('models')
cifar = require('cifar')
optnet = require('optnet')

config = {
    -- Training configuration
    batch_size = 5000,
    epochs = 50,
    cuda = true,
    model_type = 'caffe',

    -- Data configuration
    nclass = 10,
    train_cnt = 40000,
    validate_cnt = 10000,
    test_cnt = 10000
}

sgd_params = {
	learningRate = 1e-2,
	--weightDecay = 1e-3
}

torch.manualSeed(1)
--------------------------------------------------------------------------------
-- 1. Data load and Normalization
--------------------------------------------------------------------------------
print(string.format('Loading cifar-%d data...', config.nclass))
cifar_data = cifar.load(config.nclass)
train_data, test_data = cifar_data.train, cifar_data.test

train_set = {
    data = train_data.data[{{1, config.train_cnt}}],
    labels = train_data.labels[{{1, config.train_cnt}}]
}

validate_set = {
    data = train_data.data[{{config.train_cnt+1, config.train_cnt+config.validate_cnt}}],
    labels = train_data.labels[{{config.train_cnt+1, config.train_cnt+config.validate_cnt}}]
}

test_set = {
    data = test_data.data[{{1, config.test_cnt}}],
    labels = test_data.labels[{{1, config.test_cnt}}]
}
print('Loaded')
print('')

local function get_data_set_size(data_set)
    return data_set.labels:size(1)
end

--------------------------------------------------------------------------------
-- 2. Model Define
--------------------------------------------------------------------------------
cutorch.setDevice(2)
criterion = nn.ClassNLLCriterion()
model = models.load(config.model_type, 10)
model:add(nn.LogSoftMax())

if config.cuda then
	require 'cunn'
	model:cuda()
	criterion:cuda()
	train_set.data = train_set.data:cuda()
	train_set.labels = train_set.labels:cuda()
	validate_set.data = validate_set.data:cuda()
	validate_set.labels = validate_set.labels:cuda()
	test_set.data = test_set.data:cuda()
	test_set.labels = test_set.labels:cuda()
end

print('Network to train:')
print(model)
print()

print('Memory Usage:')
print(optnet.countUsedMemory(model))
print()

--------------------------------------------------------------------------------
-- 3.1 Trainer Define
--------------------------------------------------------------------------------
x, dl_dx = model:getParameters()

function step(batch_size)
	local loss_cur = 0
	local cnt = 0
	local shuffle = torch.randperm(config.train_cnt)
	batch_size = batch_size or 500

	for t = 1, config.train_cnt, batch_size do
		local size = math.min(t + batch_size - 1, config.train_cnt) - t
		local inputs = torch.Tensor(size, 3, 32, 32)
		local targets = torch.Tensor(size)

		if config.cuda then
			inputs = inputs:cuda()
			targets = targets:cuda()
		end

		for i = 1, size do
			local input = train_set.data[shuffle[i+t]]
			local target = train_set.labels[shuffle[i+t]]
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

		_, fs = optim.sgd(feval, x, sgd_params)
		cnt = cnt + 1
		loss_cur = loss_cur + fs[1]
	end

	return loss_cur / cnt
end

function evaluation(data_set, batch_size)
	local cnt_correct = 0
    local data_set_size = get_data_set_size(data_set)
	batch_size = batch_size or 500

	for i = 1, data_set_size, batch_size do
		local size = math.min(i + batch_size - 1, data_set_size) - i + 1
		local inputs = data_set.data[{ {i,i+size-1} }]
		local targets = data_set.labels[{ {i,i+size-1} }]

        if config.cuda then
            targets = targets:cudaLong()
        end

		local outputs = model:forward(inputs)
		local _, indices = torch.max(outputs, 2)
		local cnt_right = indices:eq(targets):sum()
	
		cnt_correct = cnt_correct + cnt_right
	end

	return cnt_correct / data_set_size
end


--------------------------------------------------------------------------------
-- 3.2 Train Model
--------------------------------------------------------------------------------
print('\n------------------------------------------------')
print('* Training')

local loss_table_train = {}
local acc_table_train = {}
local acc_table_valid = {}

model:training()
for i = 1, config.epochs do
	print(string.format('Epoch %d,', i))
	local loss = step(config.batchsize)
	print(string.format('	Train loss		: %.8f', loss))
	local acc_train = evaluation(train_set, config.batch_size)
	print(string.format('	Train Accuracy		: %.8f', acc_train*100))
	local acc_valid = evaluation(validate_set, config.batch_size)
	print(string.format('	Validation Accuracy	: %.8f', acc_valid*100))
	
	table.insert(loss_table_train, loss)
	table.insert(acc_table_train, acc_train*100)
	table.insert(acc_table_valid, acc_valid*100)
end

print('Memory Usage:')
print(optnet.countUsedMemory(model))
print()

print('Evaluation')
local acc_test = evaluation(test_set)
print(string.format('Test Accuracy: %.8f', acc_test*100))
