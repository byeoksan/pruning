----------------------------------------------------------------
-- MNIST by Lenet-5 based on 'small dnn example by Rudra Poudel'
----------------------------------------------------------------
-- Iterative pruning version
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'LinearWithMask'
require 'SpatialConvolutionWithMask'
require 'xlua'
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
cmd:option('-learningRatePre',	1e-1,		'learning for pretraining rate at t=0')
cmd:option('-learningRateRe',	1e-2*2,		'learning for retraining rate at t=0')
cmd:option('-weightDecay',		1e-2,		'weight decay')
cmd:option('-momentum',			0.6,		'momentum')
cmd:option('-batchsize',		200,		'mini-batch size (1 = pure stochastic)')
cmd:option('-iterPretrain',		10,			'Maximum iteration number for Pretraining')
cmd:option('-iterRetrain',		10,			'Maximum iteration number for Retraining')
cmd:option('-iterPruning',		5,			'The number of iterative pruning')
cmd:text()

print("\n... learning setup")
opt = cmd:parse(arg or {})
print(opt)

function memoryUse()
	local free, total = cutorch.getMemoryUsage(opt.gpuid)
	print(string.format('... %.4fMB is available out of %.4fMB (%.2f%%)', free/1024/1024, total/1024/1024, free/total*100))
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
	learningRate = opt.learningRatePre,
	weightDecay = opt.weight_decay,
	momentum = opt.momentum,
	learningRateDecay = 5e-7
}
local optim_method = optim.sgd

local filename = {
	NumWeights = string.format(tostring(opt.model)..'NumWeight.mat'),
	WeightPretrain = string.format(tostring(opt.model)..'WeightPretrain.mat'),
	WeightPruned = string.format(tostring(opt.model)..'WeightPruned.mat'),
	WeightRetrain = string.format(tostring(opt.model)..'WeightRetrained.mat'),
	ModelPretrain = string.format(tostring(opt.model)..'ModelPretrain.t7'),
	ModelRetrain = string.format(tostring(opt.model)..'ModelRetrain.t7'),
	AccLoss = string.format(tostring(opt.model)..'AccLoss.mat')
}

------------------------------------------------------------------------
-- Defining DNN model
------------------------------------------------------------------------
model = nn.Sequential()

if opt.model == 'lenet' then					-- LeNet-5
	model:add(nn.View(1,28,28))
	model:add(SpatialConvolutionWithMask(1,6,5,5))	-- (1x28x28) goes in, (6x24x24) goes out -- Conv1
	model:add(nn.SpatialMaxPooling(2,2,2,2)) 	-- (6x24x24) goes out, (6x12x12) goes out
	model:add(nn.ReLU())
	model:add(SpatialConvolutionWithMask(6,16,3,3))	-- (6x12x12) goes in, (16x10x10) goes out -- Conv2
	model:add(nn.SpatialMaxPooling(2,2,2,2))	-- (16x10x10) goes in, (16x5x5) goes out
	model:add(nn.ReLU())
	model:add(nn.View(16*5*5))
	model:add(LinearWithMask(16*5*5, 120))
	model:add(nn.ReLU())
	model:add(LinearWithMask(120,84))
	model:add(nn.ReLU())
	model:add(LinearWithMask(84,10))
	model:add(nn.LogSoftMax())
else
	model:add(nn.View(1,28,28))
	model:add(nn.SpatialConvolution(1,6,5,5))	-- (1x28x28) goes in, (6x24x24) goes out -- Conv1
	model:add(nn.SpatialMaxPooling(2,2,2,2)) 	-- (6x24x24) goes out, (6x12x12) goes out
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(6,3,3,3))	-- (6x12x12) goes in, (3x10x10) goes out -- Conv2
	model:add(nn.SpatialMaxPooling(2,2,2,2))	-- (3x10x10) goes in, (3x5x5) goes out
	model:add(nn.ReLU())
	model:add(nn.View(3*5*5))

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
	trainset.data = trainset.data:cuda()
	trainset.label = trainset.label:cuda()
	validationset.data = validationset.data:cuda()
	validationset.label = validationset.label:cuda()
	testset.data = testset.data:cuda()
	testset.label = testset.label:cuda()
end


-- Trainer
-- Tensor variables for model params and gradient params
local x, dl_dx = model:getParameters()
function step(batch_size, phase)
	local loss_cur = 0
	local cnt = 0
	local shuffle = torch.randperm(sizeparam.tr)
	batch_size = batch_size or 500

	model:training()
	for t = 1, sizeparam.tr, batch_size do
		xlua.progress(t, sizeparam.tr)
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

		if		phase == 'Pre'	then	optim_state.learningRate = opt.learningRatePre
		elseif 	phase == 'Re'	then	optim_state.learningRate = opt.learningRateRe	end
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
		xlua.progress(i, dataset.size)
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


-------------------------------
-- PreTrain
-------------------------------
print('\n------------------------------------------------')
print('\n>> Pretraining Phase')
LossTablePretrain = {}
AccTableTrainPretrain = {}
AccTableValidPretrain = {}

last_acc_valid = 0
decreasing = 0
threshold = 2

local timePretrain = 0
for i = 1, opt.iterPretrain do
	local time = sys.clock()
	local loss = step(opt.batch_size, 'Pre')
	local acc_train = evaluation(trainset, opt.batch_size)
	local acc_valid = evaluation(validationset, opt.batch_size)
	print(string.format('\n>> Epoch %d,', i))
	print(string.format('... Train loss				: %.8f', loss))
	print(string.format('... Train Accuracy			: %.8f', acc_train*100))
	print(string.format('... Validation Accuracy	: %.8f', acc_valid*100))
	memoryUse()

	table.insert(LossTablePretrain, loss)
	table.insert(AccTableTrainPretrain, acc_train*100)
	table.insert(AccTableValidPretrain, acc_valid*100)
	
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

	time = sys.clock() - time
	timePretrain = timePretrain + time
	print('... time consumed at epoch ' .. i ..' = ' .. (time) .. 's')
end

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
	fc2 = torch.cat(weight[7]:view(10*30), weight[8], 1):double()
end

-- Save pretraining data
matio.save(filename.WeightPretrain, {Conv1 = conv1, Conv2 = conv2, Fc1 = fc1, Fc2 = fc2, Fc3 = fc3})
torch.save(filename.ModelPretrain, model)

print('... time consumed for Pretraining = ' .. (timePretrain) .. 's')
print("... counting memory after Pretraining")
print(optnet.countUsedMemory(model))
print('\n>> Pretraining Phase, Evaluation')
local AccTestPretrain = evaluation(testset)
print(string.format('\n... Test Accuracy after Pretraining	: %.4f', AccTestPretrain*100))


-------------------------------
-- Iterative Pruning
-------------------------------
local qConv = 1
local qFc = 1
numWeight = {}
ConvLayerSet = model:findModules('SpatialConvolutionWithMask')
FcLayerSet = model:findModules('LinearWithMask')
local numWeightPre = 0
for i = 1, #ConvLayerSet	do	numWeightPre = numWeightPre + ConvLayerSet[i].weight:nElement()	end
for i = 1, #FcLayerSet 		do	numWeightPre = numWeightPre + FcLayerSet[i].weight:nElement()	end
AccTableTestRetrain = {}
for j = 1, opt.iterPruning do
	-------------------------------------------------------------
	-- Pruning
	-------------------------------------------------------------
	print('\n------------------------------------------------')
	print('\n>> Pruning #' ... j)

	-- Set thresholds and do pruning
	for i = 1, #ConvLayerSet do
		--qConv[#qConv+1] = 0.5
		--ConvLayerSet[i]:prune(qConv[i])
		qConv = qConv * 1.1
		ConvLayerSet[i]:prune(qConv)
	end
	for i = 1, #FcLayerSet do
		--qFc[#qFc+1] = 0.5
		--FcLayerSet[i]:prune(qFc[i])
		qFc = qFc * 1.1
		FcLayerSet[i]:prune(qFc)
	end

	-- Count # of pruned weights
	local numPrunedConv = {}
	local numPrunedConvTotal = 0
	local numPrunedFc = {}
	local numPrunedFcTotal = 0

	for i, m in ipairs(model:findModules('SpatialConvolutionWithMask')) do
		numPrunedConv[#numPrunedConv+1] = m.weightMask:eq(0):sum()
		numPrunedConvTotal = numPrunedConvTotal + numPrunedConv[#numPrunedConv] 
	end
	for i, m in ipairs(model:findModules('LinearWithMask')) do
		numPrunedFc[#numPrunedFc+1] = m.weightMask:eq(0):sum()
		numPrunedFcTotal = numPrunedFcTotal + numPrunedFc[#numPrunedFc] 
	end
	local wt = numWeightPre - numPrunedConvTotal - numPrunedFcTotal
	print('... # of pruned synapses in convolution layers: ' .. numPrunedConvTotal)
	print('... # of pruned synapses in fully-connected layers: ' .. numPrunedFcTotal)
	print('... # of remaining synapses in total: ' .. wt)
	print('... (# of synapses in original network: ' .. numWeightPre .. ')')
	table.insert(numWeight, wt)

	-------------------------------------------------------------
	-- ReTraining
	-------------------------------------------------------------
	print('\n------------------------------------------------')
	print('\n>> Retraining')

	local timeRetrain = 0
	for i = 1, opt.iterRetrain do
		local time = sys.clock()
		local loss = step(opt.batch_size, 'Re')
		local acc_train = evaluation(trainset, opt.batch_size)
		local acc_valid = evaluation(validationset, opt.batch_size)
		print(string.format('\nEpoch %d,', i))
		print(string.format('	Train loss		: %.8f', loss))
		print(string.format('	Train Accuracy		: %.8f', acc_train*100))
		print(string.format('	Validation Accuracy	: %.8f', acc_valid*100))
		memoryUse()
		
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

		time = sys.clock() - time
		timeRetrain = timeRetrain + time
		print('... time consumed at epoch ' .. i ..' = ' .. (time) .. 's')
	end

	local AccTestRetrain = evaluation(testset)
	print(string.format('\n... Test Accuracy after Retraining	: %.4f', AccTestRetrain*100))
	table.insert(AccTableTestRetrain, AccTestRetrain*100)
end


------------------------------------------------------------------------
-- Saving models and data
------------------------------------------------------------------------
--wP, _ = model:parameters()
-- Count # of pruned weights
numPrunedConv = {}
numPrunedConvTotal = 0
numPrunedFc = {}
numPrunedFcTotal = 0

for i, m in ipairs(model:findModules('SpatialConvolutionWithMask')) do
	numPrunedConv[#numPrunedConv+1] = m.weightMask:eq(0):sum()
	numPrunedConvTotal = numPrunedConvTotal + numPrunedConv[#numPrunedConv] 
end
for i, m in ipairs(model:findModules('LinearWithMask')) do
	numPrunedFc[#numPrunedFc+1] = m.weightMask:eq(0):sum()
	numPrunedFcTotal = numPrunedFcTotal + numPrunedFc[#numPrunedFc] 
end

local wt = numWeightPre - numPrunedConvTotal - numPrunedFcTotal
print('... # of pruned synapses in convolution layers: ' .. numPrunedConvTotal)
print('... # of pruned synapses in fully-connected layers: ' .. numPrunedFcTotal)
print('... # of remaining synapses in total: ' .. wt)
print('... (# of synapses in original network: ' .. numWeightPre .. ')')

-- Store only pruned weights
prunedConvLayerSet = {}
prunedFcLayerSet = {}
for i = 1, #ConvLayerSet do
	local wP = ConvLayerSet[i].weight:reshape(ConvLayerSet[i].weight:nElement())
	local _, idx = torch.abs(wP):sort()
	idx = idx[{{numPrunedConv[i]+1, -1}}]
	local lal = torch.ByteTensor(wP:size(1)):fill(0)
	for j = 1, idx:size(1) do
		lal[idx[j]] = 1
	end
	prunedConvLayerSet[#prunedConvLayerSet+1] = wP[lal]
end
for i = 1, #FcLayerSet do
	local wP = FcLayerSet[i].weight:reshape(FcLayerSet[i].weight:nElement())
	local _, idx = torch.abs(wP):sort()
	idx = idx[{{numPrunedFc[i]+1, -1}}]
	local lal = torch.ByteTensor(wP:size(1)):fill(0)
	for j = 1, idx:size(1) do
		lal[idx[j]] = 1
	end
	prunedFcLayerSet[#prunedFcLayerSet+1] = wP[lal]
end

-- Save files
print('\n------------------------------------------------')
print('>> Model, Weight, Accuracy, and Loss tensors are saved.')

torch.save(filename.ModelRetrain, model)
matio.save(filename.NumWeights, {numWeight = torch.Tensor(numWeight)})
matio.save(filename.WeightRetrain, {Conv1 = prunedConvLayerSet[1]:double(), 
									Conv2 = prunedConvLayerSet[2]:double(), 
									Fc1 = prunedFcLayerSet[1]:double(), 
									Fc2 = prunedFcLayerSet[2]:double(), 
									Fc3 = prunedFcLayerSet[3]:double()} )
matio.save(filename.AccLoss, {
								LossPretrain = torch.Tensor(LossTablePretrain),
								AccTrainPretrain = torch.Tensor(AccTableTrainPretrain),
								AccValidPretrain = torch.Tensor(AccTableValidPretrain),
								AccTestPretrain = AccTestPretrain*100,
								AccTestRetrain = torch.Tensor(AccTableTestRetrain)
								})
