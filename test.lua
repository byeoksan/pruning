#!/usr/bin/env th

local progress = require('progress')
local dataset = require('dataset')
local util = require('util')

local M = {}

function M.evaluate(model, data, labels, config_params)
    model:evaluate()
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

    local cmd = util.createModelCmdLine(false)
    cmd = util.mergeCmdLineOptions(cmd, util.createConfigCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, dataset.createDataCmdLine())

    local parsed = cmd:parse(arg or {})

    local model_params = util.parsedCmdLineToModelParams(parsed)
    local config_params = util.parsedCmdLineToConfigParams(parsed)
    local data_params = dataset.parsedCmdLineToDataParams(parsed)

    if not model_params.model then
        io.stderr:write(cmd:help())
        io.stderr:write('\n')
        return
    end
    model_params.model:add(nn.LogSoftMax())

    if config_params.debug then
        print('=== Testing Parameters ===')
        print(config_params)
        print('=== Model ===')
        print(model_params.model)
        print('=== Test Data ===')
        print(data_params.data)
    end

    -- Cuda-ify
    if config_params.cuda then
        require 'cunn'
        model_params.model:cuda()
        data_params.data.train.data = data_params.data.train.data:cuda()
        data_params.data.train.labels = data_params.data.train.labels:cuda()
        data_params.data.test.data = data_params.data.test.data:cuda()
        data_params.data.test.labels = data_params.data.test.labels:cuda()
    end

    local train_accuracy = M.evaluate(model_params.model, data_params.data.train.data, data_params.data.train.labels, config_params)
    print(string.format('Train Accuracy: %f', train_accuracy * 100))
    local validate_accuracy = M.evaluate(model_params.model, data_params.data.validate.data, data_params.data.validate.labels, config_params)
    print(string.format('validate Accuracy: %f', validate_accuracy * 100))
    local test_accuracy = M.evaluate(model_params.model, data_params.data.test.data, data_params.data.test.labels, config_params)
    print(string.format('Test Accuracy: %f', test_accuracy * 100))
end

return M
