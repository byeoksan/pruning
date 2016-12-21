#!/usr/bin/env th

local M = {}

--[[ CIFAR data directory structure
cifar/
├── cifar-100-t7/
│   ├── data_batch_1.t7
│   ├── data_batch_2.t7
│   ├── data_batch_3.t7
│   ├── data_batch_4.t7
│   ├── data_batch_5.t7
│   ├── meta.t7
│   └── test_batch.t7
└── cifar-10-t7/
│   ├── data_batch_1.t7
│   ├── data_batch_2.t7
│   ├── data_batch_3.t7
│   ├── data_batch_4.t7
│   ├── data_batch_5.t7
│   ├── meta.t7
│   └── test_batch.t7
└── cifar-5-t7/
    ├── data_batch_1.t7
    ├── data_batch_2.t7
    ├── data_batch_3.t7
    ├── data_batch_4.t7
    ├── data_batch_5.t7
    ├── meta.t7
    └── test_batch.t7
--]]
local function _load_cifar(nclass)
    -- Maximum 50000 data
    local train_data = {
        data = torch.DoubleTensor(50000, 3, 32, 32),
        labels = torch.DoubleTensor(50000)
    }

    -- Maximum 10000 data
    local test_data = {
        data = torch.DoubleTensor(10000, 3, 32, 32),
        labels = torch.DoubleTensor(10000)
    }

    local data = 'data'
    local labels
    if nclass == 100 then
        labels = 'fine_labels'
    else
        labels = 'labels'
    end

    local train_size = 0
    for i = 1, 5 do
        local filename = string.format('cifar/cifar-%d-t7/data_batch_%d.t7', nclass, i)
        local batch = torch.load(filename)
        local batch_size = batch[data]:size(1)

        train_data.data[{{train_size+1, train_size+batch_size}}] = batch[data]:reshape(batch_size, 3, 32, 32)
        train_data.labels[{{train_size+1, train_size+batch_size}}] = batch[labels] + 1

        train_size = train_size + batch_size
    end

    -- Load test data
    local test_batch = torch.load(string.format('cifar/cifar-%d-t7/test_batch.t7', nclass))
    local test_size = test_batch[data]:size(1)
    test_data.data[{{1, test_size}}] = test_batch[data]:reshape(test_size, 3, 32, 32)
    test_data.labels[{{1, test_size}}] = test_batch[labels] + 1

    train_data.data = train_data.data[{{1, train_size}}]
    train_data.labels = train_data.labels[{{1, train_size}}]
    test_data.data = test_data.data[{{1, test_size}}]
    test_data.labels = test_data.labels[{{1, test_size}}]

    -- Normalize
    mean = torch.mean(train_data.data, 1):squeeze()
    for i = 1, train_size do
        train_data.data[i]:csub(mean)
    end
    for i = 1, test_size do
        test_data.data[i]:csub(mean)
    end

 	data = {
        train = {
			data = torch.DoubleTensor(train_size, 3, 32, 32),
        	labels = torch.DoubleTensor(train_size)
		},
        validate = {
			data = torch.DoubleTensor(test_size, 3, 32, 32),
        	labels = torch.DoubleTensor(test_size)
		},
        test = {
			data = torch.DoubleTensor(test_size, 3, 32, 32),
        	labels = torch.DoubleTensor(test_size)
		},
    }

	data.train.data:copy(train_data.data)
	data.train.labels:copy(train_data.labels)
	-- TODO for now, just use test data as validate data
	data.validate.data:copy(test_data.data)
	data.validate.labels:copy(test_data.labels)
	data.test.data:copy(test_data.data)
	data.test.labels:copy(test_data.labels)

	return data
end

local function _load_mnist()
    local mnist = require('mnist')
    local data = {
        train = {},
        validate = {},
        test = {},
    }

    local d = mnist.traindataset()
    local shape = d.data:size()

    shape[1] = 50000
    data.train.data = torch.DoubleTensor(shape)
    data.train.labels = torch.DoubleTensor(50000)
    data.train.data:copy(d.data[{{1, 50000}}])
    data.train.labels:copy(d.label[{{1, 50000}}])

    shape[1] = 10000
    data.validate.data = torch.DoubleTensor(shape)
    data.validate.labels = torch.DoubleTensor(10000)
    data.validate.data:copy(d.data[{{50001, 60000}}])
    data.validate.labels:copy(d.label[{{50001, 60000}}])

    d = mnist.testdataset()
    shape = d.data:size()

    data.test.data = torch.DoubleTensor(shape)
    data.test.labels = torch.DoubleTensor(shape[1])
    data.test.data:copy(d.data)
    data.test.labels:copy(d.label)

    -- Normalize
    local mean = data.train.data:mean()
    local std = data.train.data:std()
    data.train.data:add(-mean)
    data.validate.data:add(-mean)
    data.test.data:add(-mean)

    data.train.data:div(std)
    data.validate.data:div(std)
    data.test.data:div(std)

    -- Label adjust
    data.train.labels:add(1)
    data.validate.labels:add(1)
    data.test.labels:add(1)

    return data
end

local function _load_cifar5()
    return _load_cifar(5)
end

local function _load_cifar10()
    return _load_cifar(10)
end

local function _load_cifar100()
    return _load_cifar(100)
end

local load_functions = OrderedMap({
    {mnist = _load_mnist},
    {cifar5 = _load_cifar5},
    {cifar10 = _load_cifar10},
    {cifar100 = _load_cifar100},
})

function M.createDataCmdLine()
    local cmd = torch.CmdLine()
    cmd:option('-data', load_functions:keys()[1], string.format('Dataset to use (%s)', tostring(load_functions:keys()):sub(2, -2):gsub(',', ', ')))
    return cmd
end

function M.parsedCmdLineToDataParams(parsed)
    return {
        data = M.load(parsed.data)
    }
end

function M.load(data)
    return load_functions[data]()
end

return M
