#!/usr/bin/env th

local progress = require('progress')
local dataset = require('dataset')

local M = {}

function M.evaluate(model, data, labels, config_params)
    model:evaluate()
    config_params = config_params or {}
    local shape = data:size()
    local data_size = shape[1]
    local batch = config_params.batch or 128

    local correct = 0
    for i = 1, data_size, batch do
        local mini_batch = math.min(i+batch-1, data_size) - i + 1
        if config_params.progress then
            progress.progress(i, data_size)
        end

        shape[1] = mini_batch
        local inputs = torch.DoubleTensor(shape)
        local targets = torch.LongTensor(mini_batch)

        if config_params.cuda then
            inputs = inputs:cuda()
            targets = targets:cudaLong()
        end

        inputs:copy(data[{{i, i+mini_batch-1}}])
        targets:copy(labels[{{i, i+mini_batch-1}}])

        local output = model:forward(inputs)
        local _, indices = torch.max(output, 2)
        local curr_correct = indices:eq(targets):sum()

        correct = correct + curr_correct
    end
    progress.clear()

    return correct / data_size
end

function M.main(arg)
    -- arg: command line arguments
    local models = require('models')

    local cmd = torch.CmdLine()
    cmd:option('-model', '', 'Trained model to test')
	cmd:option('-nClass', 0, 'Number of class')
    cmd:option('-cuda', false, 'Whether to use cuda')
    cmd:option('-batch', 128, 'Batch size')
    cmd:option('-progress', false, 'True to show progress')
    cmd:option('-debug', false, 'True for debugging')

    local params = cmd:parse(arg or {})

    -- Load the model
    local model
    if params.model == '' then
        io.stderr:write(cmd:help())
        io.stderr:write('\n')
        return
    else
        model = models.restore(params.model)
    end

    model:add(nn.LogSoftMax())
    config_params = {
        cuda = params.cuda,
        batch = params.batch,
        progress = params.progress,
        debug = params.debug,
    }

    if config_params.debug then
        print('=== Testing Parameters ===')
        print(config_params)
        print('=== Model ===')
        print(model)
    end

	if params.nClass ~= 0 then
		local data = dataset.load_cifar(params.nClass)
	else
    	local data = dataset.load()
	end
    if config_params.debug then
        print(data)
    end

    -- Cuda-ify
    if params.cuda then
        require 'cunn'
        model:cuda()
        data.train.data = data.train.data:cuda()
        data.train.labels = data.train.labels:cuda()
        data.test.data = data.test.data:cuda()
        data.test.labels = data.test.labels:cuda()
    end

    local train_accuracy = M.evaluate(model, data.train.data, data.train.labels, config_params)
    print(string.format('Train Accuracy: %f', train_accuracy * 100))
    local validate_accuracy = M.evaluate(model, data.validate.data, data.validate.labels, config_params)
    print(string.format('validate Accuracy: %f', validate_accuracy * 100))
    local test_accuracy = M.evaluate(model, data.test.data, data.test.labels, config_params)
    print(string.format('Test Accuracy: %f', test_accuracy * 100))
end

return M
