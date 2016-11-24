#!/usr/bin/env th

require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'

models = require('models')
cifar = require('cifar')
optnet = require('optnet')

config = {
    -- Training configuration
    batch_size = 50,
    epochs = 50,
    cuda = false,
    model_type = 'allcnn',

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

io.write('Now train...\n')
local loss_table_train = {}
local acc_table_train = {}
local acc_table_valid = {}

model:training()
for i = 1, config.epochs do
	io.write(string.format('Epoch %d\n', i))
	local loss = step()
	io.write(string.format('\t         Train loss: %.8f\n', loss))
	local acc_train = evaluation(train_set)
	io.write(string.format('\t     Train Accuracy: %.8f\n', acc_train*100))
	local acc_valid = evaluation(validate_set)
	io.write(string.format('\tValidation Accuracy: %.8f\n', acc_valid*100))
	
	gnuplot.figure(1)
	gnuplot.xlabel('weight')
	gnuplot.title('Weight Distirubtion')

	table.insert(loss_table_train, loss)
	table.insert(acc_table_train, acc_train*100)
	table.insert(acc_table_valid, acc_valid*100)

	-- Plot weight distribution
	local w_temp, _ = model:parameters()
	local w_acc = w_temp[1]:view(w_temp[1]:nElement())
	for j = 2, #w_temp do
		w_acc = torch.cat(w_acc, w_temp[j]:view(w_temp[j]:nElement()), 1)
	end
	w_temp = nil  -- necessary because of out of memory issue

	-- Find nonzero weights
	local epsilon = 1e-2*5
	local cnt = 0
	local w_nz = {}
	for j = 1, w_acc:size(1) do
		--if j%1000 == 0 then print(w_acc[j]) end
		if w_acc[j] > epsilon or w_acc[j] < -epsilon then
			w_nz[#w_nz+1] = w_acc[j]
		end
	end
	w_acc = nil
	w_nz = torch.Tensor(w_nz)
	print(w_nz:size())
	print(w_nz:std())
	gnuplot.figure(1)
	gnuplot.xlabel('weights')
	gnuplot.hist(w_nz, 1000)
end

io.write('Memory Usage:', tostring(optnet.countUsedMemory(model)), '\n')
io.write('Test result:\n')
local acc_test = evaluation(test_set)
io.write(string.format('Test Accuracy: %.8f\n', acc_test*100))


