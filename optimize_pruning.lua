#!/usr/bin/env th

require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'
require 'SpatialConvolutionWithMask'

models = require('models')
cifar = require('cifar')
optnet = require('optnet')

config = {
    -- Training configuration
    batch_size = 50,
    epochsPre = 50,				-- learning epochs before pruning(pretraining)
	epochsPost = 100,			-- learning epochs after pruning(posttraining)
    cuda = false,
    model_type = 'allcnnPruning',

    -- Data configuration
    nclass = 10,
    train_cnt = 300,
    validate_cnt = 50,
    test_cnt = 50
}

sgd_config = {
	learningRate = 1e-2,
	--weightDecay = 1e-3
}

------------------- Data -------------------
io.write(string.format('Loading cifar-%d data...', config.nclass))
io.flush()
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
io.write('Loaded\n')

local function get_data_set_size(data_set)
    return data_set.labels:size(1)
end

------------------- Model -------------------
model = models.load(config.model_type, 10)
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

if config.cuda then
	require 'cunn'
	require 'cutorch'
	model:cuda()
	criterion:cuda()
	train_set.data = train_set.data:cuda()
	train_set.labels = train_set.labels:cuda()
	validate_set.data = validate_set.data:cuda()
	validate_set.labels = validate_set.labels:cuda()
	test_set.data = test_set.data:cuda()
	test_set.labels = test_set.labels:cuda()
end

w, dl_dw = model:getParameters()

io.write('Network to train:\n', tostring(model), '\n')
io.write('Memory Usage:\n', tostring(optnet.countUsedMemory(model)), '\n')

function step()
	local loss_cur = 0
	local cnt = 0
	local shuffle = torch.randperm(config.train_cnt)

	for t = 1, config.train_cnt, config.batch_size do
		local size = math.min(t + config.batch_size - 1, config.train_cnt) - t + 1
		local inputs = torch.DoubleTensor(size, 3, 32, 32)
		local targets = torch.DoubleTensor(size)

		if config.cuda then
			inputs = inputs:cuda()
			targets = targets:cuda()
		end

		for i = 1, size do
			inputs[i] = train_set.data[shuffle[i+t-1]]
			targets[i] = train_set.labels[shuffle[i+t-1]]
		end

		local feval = function(w)
            dl_dw:zero()
			local loss = criterion:forward(model:forward(inputs), targets)
			model:backward(inputs, criterion:backward(model.output, targets))
			return loss, dl_dw
		end

		_, fs = optim.sgd(feval, w, sgd_config)
		cnt = cnt + 1
		loss_cur = loss_cur + fs[1]
	end

	return loss_cur / cnt
end

function evaluation(data_set)
	local cnt_correct = 0
    local data_set_size = get_data_set_size(data_set)

	model:evaluate()
	for i = 1, data_set_size, config.batch_size do
		local size = math.min(i + config.batch_size - 1, data_set_size) - i + 1
		local inputs = data_set.data[{{i,i+size-1}}]
		local targets = data_set.labels[{{i,i+size-1}}]

        if config.cuda then
            inputs = inputs:cuda()
            targets = targets:cudaLong()
        end

		local outputs = model:forward(inputs)
		local _, indices = torch.max(outputs, 2)
		--local cnt_right = indices:eq(targets):sum()	-- use this one with cuda
		local cnt_right = indices:eq(targets:long()):sum()	-- use this one w/o cuda

		cnt_correct = cnt_correct + cnt_right
	end

	return cnt_correct / data_set_size
end

function plotWeight(w, numFigure, mode)
	local w_acc = w[1]:view(w[1]:nElement())
	for j = 2, #w do
		w_acc = torch.cat(w_acc, w[j]:view(w[j]:nElement()), 1)
	end
	gnuplot.figure(numFigure)
	gnuplot.xlabel('weights')
	if mode == 'pre' then		gnuplot.title('Pre-training to learn connectivity')
	elseif mode == 'post' then	gnuplot.title('Post-training after pruning')
	end
	gnuplot.hist(w_acc, 1000)
end

function stdWeight(w)
	local w_acc = w[1]:view(w[1]:nElement())
	for j = 2, #w do
		w_acc = torch.cat(w_acc, w[j]:view(w[j]:nElement()), 1)
	end
	return w_acc:std()
end

io.write('Now train...\n')
local loss_table_train = {}
local acc_table_train = {}
local acc_table_valid = {}


-- pretraining to learn connectivity
model:training()
for i = 1, config.epochsPre do
	io.write(string.format('Epoch %d\n', i))
	local loss = step()
	io.write(string.format('\t         Train loss: %.8f\n', loss))
	local acc_train = evaluation(train_set)
	io.write(string.format('\t     Train Accuracy: %.8f\n', acc_train*100))
	local acc_valid = evaluation(validate_set)
	io.write(string.format('\tValidation Accuracy: %.8f\n', acc_valid*100))

	table.insert(loss_table_train, loss)
	table.insert(acc_table_train, acc_train*100)
	table.insert(acc_table_valid, acc_valid*100)

	-- plot weight distribution every epoch
	local wTemp, _ = model:parameters()
	print(plotWeight(wTemp, 1, 'pre'))
	wTemp = nil
end

--[[
-- plot weight distribution after pretraining
local wTemp, _ = model:parameters()
plotWeight(wTemp, 1, 'pre')
wTemp = nil
]]--

-- pruning
wTemp, _ = model:parameters()
local wStd = stdWeight(wTemp)
wTemp = nil
local qParam = 0.7 -- quality parameter


-- post-training 
model:training()
for i = 1, config.epochsPost do
	io.write(string.format('Epoch %d\n', i))
	local loss = step()
	io.write(string.format('\t         Train loss: %.8f\n', loss))
	local acc_train = evaluation(train_set)
	io.write(string.format('\t     Train Accuracy: %.8f\n', acc_train*100))
	local acc_valid = evaluation(validate_set)
	io.write(string.format('\tValidation Accuracy: %.8f\n', acc_valid*100))

	table.insert(loss_table_train, loss)
	table.insert(acc_table_train, acc_train*100)
	table.insert(acc_table_valid, acc_valid*100)
end

local wTemp, _ = model:parameters()
plotWeight(wTemp, 2, 'post')
wTemp = nil

io.write('Memory Usage:', tostring(optnet.countUsedMemory(model)), '\n')
io.write('Test result:\n')
local acc_test = evaluation(test_set)
io.write(string.format('Test Accuracy: %.8f\n', acc_test*100))


