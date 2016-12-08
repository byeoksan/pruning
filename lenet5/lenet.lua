----------------------------------------------------------------
-- MNIST by Lenet-5 based on 'small dnn example by Rudra Poudel'
----------------------------------------------------------------

require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'gnuplot'
local mnist = require 'mnist'
local optnet = require 'optnet'
local matio = require 'matio'

-- command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNN Example')
cmd:text()
cmd:text('Options:')
cmd:option('-type',				'cuda',		'type: float | cuda')
cmd:option('-model',			'lenet',	'type: LeNet5 | defaultNet')
cmd:option('-seed',				1,			'fixed input seed for repeatable experiments')
cmd:option('-learning_rate',	1e-2*10,	'learning rate at t=0')
cmd:option('-momentum',			0.6,		'momentum')
cmd:option('-weight_decay',		1e-3,		'weight decay')
cmd:option('-batch_size',		75,			'mini-batch size (1 = pure stochastic)')
cmd:option('-maxiter',			30,		'Maximum iteration number')
cmd:text()

print("\n... learning setup")
opt = cmd:parse(arg or {})
print(opt)

function memoryUse()
	local free, total = cutorch.getMemoryUsage(opt.gpuid)
	print(string.format('%.4fMB is available out of %.4fMB (%.2f%%)', free/1024/1024, total/1024/1024, free/total*100))
end

-- set options
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
	require 'cutorch'
	require 'cunn'
end
torch.manualSeed(opt.seed)

-- Set DNN params
local num_classes = 6
local batch_size = opt.batch_size

memoryUse()
local optim_state = {
	learningRate = opt.learning_rate,
	weightDecay = opt.weight_decay,
	momentum = opt.momentum,
	learningRateDecay = 5e-7
}
local optim_method = optim.sgd

------------------------------------------------------------------------
-- Defining DNN model
------------------------------------------------------------------------
model = nn.Sequential()

if opt.model == 'lenet' then					-- LeNet-5
	model:add(nn.View(1,28,28))
	model:add(nn.SpatialConvolution(1,6,5,5))	-- (1x28x28) goes in, (6x24x24) goes out -- Conv1
	model:add(nn.SpatialMaxPooling(2,2,2,2)) 	-- (6x24x24) goes out, (6x12x12) goes out
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(6,16,3,3))	-- (6x12x12) goes in, (16x10x10) goes out -- Conv2
	model:add(nn.SpatialMaxPooling(2,2,2,2))	-- (16x10x10) goes in, (16x5x5) goes out
	model:add(nn.ReLU())
	model:add(nn.View(16*5*5))
	model:add(nn.Linear(16*5*5, 120))
	model:add(nn.ReLU())
	model:add(nn.Linear(120,84))
	model:add(nn.ReLU())
	model:add(nn.Linear(84,10))
	model:add(nn.SoftMax())
else
	model:add(nn.View(1,28,28))
	model:add(nn.SpatialConvolution(1,6,5,5))	-- (1x28x28) goes in, (6x24x24) goes out -- Conv1
	model:add(nn.SpatialMaxPooling(2,2,2,2)) 	-- (6x24x24) goes out, (6x12x12) goes out
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(6,3,3,3))	-- (6x12x12) goes in, (3x10x10) goes out -- Conv2
	model:add(nn.SpatialMaxPooling(2,2,2,2))	-- (3x10x10) goes in, (3x5x5) goes out
	model:add(nn.ReLU())
	model:add(nn.View(3*5*5))

	--[[
	if opt.dropout then
		print('* dropout added')
		model:add(nn.Dropout(opt.dropout_p))
	end]]--

	model:add(nn.Dropout(opt.dropout_p))
	model:add(nn.Linear(3*5*5, 30))
	model:add(nn.ReLU())
	model:add(nn.Linear(30,10))
	model:add(nn.SoftMax())
end

criterion = nn.ClassNLLCriterion()
if opt.type == 'cuda' then
	model = model:cuda()
	criterion = criterion:cuda()
end
print(model,'\n')

print("\n... counting memory before training")
print(optnet.countUsedMemory(model))
memoryUse()


------------------------------------------------------------------------
-- Data load and normalization
------------------------------------------------------------------------
-- Create artificial data for testing
-- Note: Spatial*MM use BDHW and Spatial*CUDA use DHWB
-- bmode = 'DHWB' -- depth/channels x height x width x batch
traindataset = mnist.traindataset() -- you can access its size by 'traindataset.size'
testset = mnist.testdataset()

traindataset.data = traindataset.data:double()
mean = traindataset.data:mean()
std = traindataset.data:std()
train_zm = traindataset.data:add(-mean) -- mean subtraction
train_nr = train_zm:div(std) -- std scaling

local sizeparam = {
	tr = 50000,
	valid = 10000,
	test = 10000
}

-- Split traindataset to train/valid data
trainset = {
    size = sizeparam.tr,
    data = train_nr[{{1,sizeparam.tr}}]:double(), 
    label = traindataset.label[{{1,sizeparam.tr}}]
}

validationset = {
    size = sizeparam.valid,
    data = train_nr[{{sizeparam.tr+1,sizeparam.tr+sizeparam.valid}}]:double(),
    label = traindataset.label[{{sizeparam.tr+1,sizeparam.tr+sizeparam.valid}}]
}

-- Do the same thing to test data set
testset.data = testset.data:double()
testset.data:add(-mean)
testset.data:div(std)


if opt.type == 'cuda' then
	print('\n... put data into gpu')
	trainset.data = trainset.data:cuda()
	trainset.label = trainset.label:cuda()
	memoryUse()

	validationset.data = validationset.data:cuda()
	validationset.label = validationset.label:cuda()
	memoryUse()

	testset.data = testset.data:cuda()
	testset.label = testset.label:cuda()
	memoryUse()
end


-- Test model
if opt.print_layers_op then
	local o = inputs
	for i = 1, #(model.modules) do
		o = model.modules[i]:forward(o)
		print(#o)
	end
end


-- Trainer
-- Tensor variables for model params and gradient params
local x, dl_dx = model:getParameters()
function step(batch_size)
	local loss_cur = 0
	local cnt = 0
	local shuffle = torch.randperm(sizeparam.tr)
	batch_size = batch_size or 500

	model:training()
	for t = 1, sizeparam.tr, batch_size do
		local size = math.min(t + batch_size - 1, sizeparam.tr) - t
		local inputs = torch.Tensor(size, 1, 28, 28)
		local targets = torch.Tensor(size)
		
		if opt.type == 'cuda' then
			inputs = inputs:cuda()
			targets = targets:cuda()
		end

		for i = 1, size do
			local input = trainset.data[shuffle[i+t]]
			local target = trainset.label[shuffle[i+t]]
			inputs[i] = input
			targets[i] = target
		end
		targets:add(1)

		local feval = function(x_new)
			if x ~= x_new then x:copy(x_new) end
			dl_dx:zero()

			local loss = criterion:forward(model:forward(inputs), targets)
			model:backward(inputs, criterion:backward(model.output, targets))

			return loss, dl_dx
		end

		_, fs = optim.sgd(feval, x, optim_state)
		cnt = cnt + 1
		loss_cur = loss_cur + fs[1]
	end

	return loss_cur / cnt
end

function evaluation(dataset, batch_size)
	local cnt_correct = 0
	batch_size = bat_size or 500
	
	model:evaluate()
	for i = 1, dataset.size, batch_size do
		local size = math.min(i + batch_size, dataset.size) - i
		local inputs = dataset.data[{ {i, i+size} }]
		local targets = dataset.label[{ {i, i+size} }]

		local outputs = model:forward(inputs)
		local _, indices = torch.max(outputs, 2)
		indices:add(-1)
		local cnt_right
		cnt_right = indices:eq(targets:cudaLong()):sum()
		cnt_correct = cnt_correct + cnt_right
	end

	return cnt_correct / dataset.size
end

--------
-- Train
--------
print('\n... training')

local loss_table_train = {}
local acc_table_train = {}
local acc_table_valid = {}

last_acc_valid = 0
decreasing = 0
threshold = 2

filename_wmat = string.format(tostring(opt.model)..'_r_%s_w.mat', tostring(opt.learning_rate))
filename_torch = string.format(tostring(opt.model)..'_r_%s.t7', tostring(opt.learning_rate))
filename_mat = string.format(tostring(opt.model)..'_r_%s.mat', tostring(opt.learning_rate))
filename_wmat = string.format(tostring(opt.model)..'_r_%s_w.mat', tostring(opt.learning_rate))

local timetotal = 0
for i = 1, opt.maxiter do
	local time = sys.clock()
	local loss = step(opt.batch_size)
	print(string.format('Epoch %d,', i))
	print(string.format('	Train loss		: %.8f', loss))
	local acc_train = evaluation(trainset, opt.batch_size)
	print(string.format('	Train Accuracy		: %.8f', acc_train*100))
	local acc_valid = evaluation(validationset, opt.batch_size)
	print(string.format('	Validation Accuracy	: %.8f', acc_valid*100))
	memoryUse()

	table.insert(loss_table_train, loss)
	table.insert(acc_table_train, acc_train*100)
	table.insert(acc_table_valid, acc_valid*100)
	
	if acc_valid < last_acc_valid then
		if decreasing > threshold then
			print('... Early Stopping!!')
			break 
		end
		decreasing = decreasing + 1
	else
		decreasing = 0
	end
	last_acc_valid = acc_valid

	weight, _ = model:parameters()

	-- convert cudatensor to double tensor to save it in .mat format
	if opt.model == 'lenet' then
		conv1 = torch.cat(weight[1]:view(6*1*5*5), weight[2], 1):double()
		conv2 = torch.cat(weight[3]:view(16*6*3*3), weight[4], 1):double()
		fc1 = torch.cat(weight[5]:view(120*400), weight[6], 1):double()
		fc2 = torch.cat(weight[7]:view(84*120), weight[8], 1):double()
		fc3 = torch.cat(weight[9]:view(10*84), weight[10], 1):double()

	else
		conv1 = torch.cat(weight[1]:view(6*1*5*5), weight[2], 1):double()
		conv2 = torch.cat(weight[3]:view(3*6*3*3), weight[4], 1):double()
		fc1 = torch.cat(weight[5]:view(30*75), weight[6], 1):double()
		fc2 = torch.cat(weight[7]:view(10.*30), weight[8], 1):double()
	end

	time = sys.clock() - time
	timetotal = timetotal + time
	print('time consumed at epoch ' .. i ..' = ' .. (time) .. 's\n')
end
matio.save(filename_wmat, {w_conv1 = conv1,	w_conv2 = conv2, w_fc1 = fc1, w_fc2 = fc2, w_fc3 = fc3})

print('time consumed for all = ' .. (timetotal) .. 's')
print("\n... counting memory after training")
print(optnet.countUsedMemory(model))
print("\n... Evaluation")
local acc_test = evaluation(testset)
print(string.format('	Test Accuracy		: %.8f', acc_test*100))


------------------------------------------------------------------------
-- Savoing the model
------------------------------------------------------------------------
print('\n------------------------------------------------')
print('5. Save the model in '..filename_torch)
print('   Save the model in '..filename_mat)
print('   Save the model in '..filename_wmat..'\n')

torch.save(filename_torch, model)
matio.save(filename_mat, {loss_train = torch.Tensor(loss_table_train),
                            acc_train = torch.Tensor(acc_table_train),
                            acc_valid = torch.Tensor(acc_table_valid),
                            acc_test = acc_test*100
                            })


