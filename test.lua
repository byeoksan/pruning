#!/usr/bin/env th

require 'xlua'

local M = {}

local function _prepare_data()
    local cifar = require('cifar')
    return cifar.load(5)
end

function M.evaluate(model, data, labels, config_params)
    model:evaluate()
    local shape = data:size()
    local data_size = shape[1]
    local batch = config_params.batch

    local correct = 0
    for i = 1, data_size, batch do
        local mini_batch = math.min(i+batch-1, data_size) - i + 1
        if config_params.progress then
            xlua.progress(i, data_size)
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

    return correct / data_size
end

function M.main(arg)
    -- arg: command line arguments
    local models = require('models')

    local cmd = torch.CmdLine()
    cmd:option('-model', '', 'Trained model to test')
    cmd:option('-cuda', false, 'Whether to use cuda')
    cmd:option('-batch', 128, 'Batch size')
    cmd:option('-nclass', 10, 'Number of classes of CIFAR (5, 10, 100)')
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
        nclass = params.nclass,
        progress = params.progress,
        debug = params.debug,
    }

    if config_params.debug then
        print('=== Testing Parameters ===')
        print(config_params)
        print('=== Model ===')
        print(model)
    end

    -- Load the data
    -- TODO: Use MNIST
    -- TODO: _prepare_data is duplicate in train.lua
    local data = _prepare_data()
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
    local test_accuracy = M.evaluate(model, data.test.data, data.test.labels, config_params)
    print(string.format('Test Accuracy: %f', test_accuracy * 100))
end

return M
