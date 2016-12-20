#!/usr/bin/env th

require 'paths'
require 'nn'
require 'optim'

local util = require('util')
local progress = require('progress')
local dataset = require('dataset')
local test = require('test')

local M = {}

local function _step(model, criterion, data, optim_params, config_params)
    model:training()
    local batch = config_params.batch or 128
    local shape = data.train.data:size()
    local data_size = shape[1]
    local perm = torch.randperm(data_size):long()
    local loss = 0

    local count = 0
    for b = 1, data_size, batch do
        local mini_batch = math.min(b+batch-1, data_size) - b + 1
        if config_params.progress then
            progress.progress(b, data_size)
        end

        shape[1] = mini_batch
        local inputs = torch.DoubleTensor(shape)
        local targets = torch.DoubleTensor(mini_batch)

        if config_params.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        inputs:copy(data.train.data:index(1, perm[{{b, b+mini_batch-1}}]))
        targets:copy(data.train.labels:index(1, perm[{{b, b+mini_batch-1}}]))

        local weight, gradWeight = model:getParameters()
        gradWeight:zero()
        local eval = function(weight)
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))
            return loss, gradWeight
        end

        _, fs = optim.sgd(eval, weight, optim_params)
        loss = loss + fs[1]

        count = count + 1
    end
    progress.clear()

    return loss / count
end

function M.train(model, criterion, data, optim_params, train_params, config_params)
    local saveEpoch = train_params.saveEpoch or 0
    local saveName = train_params.saveName
    local epochs = train_params.epochs or 10

    for epoch = 1, epochs do
        print(string.format('Training Epoch %d...', epoch))
        local loss = _step(model, criterion, data, optim_params, config_params)
        print(string.format('\tTrain loss: %f', loss))

        -- Intermmediate Save
        if (saveEpoch > 0 and (epoch % saveEpoch) == 0) or epoch == epochs then
            local name = string.format('%s_%d.t7', saveName, epoch)
            if config_params.cuda then
                model:double()
                torch.save(name, model)
                model:cuda()
            else
                torch.save(name, model)
            end
        end

        -- Intermmediate Report
        local train_accuracy = test.evaluate(model, data.train.data, data.train.labels, config_params)
        print(string.format('\tTrain Accuracy: %f', train_accuracy * 100))
        local validate_accuracy = test.evaluate(model, data.validate.data, data.validate.labels, config_params)
        print(string.format('\tValidate Accuracy: %f', validate_accuracy * 100))
        local test_accuracy = test.evaluate(model, data.test.data, data.test.labels, config_params)
        print(string.format('\tTest Accuracy: %f', test_accuracy * 100))
    end
end

function M.createTrainCmdLine()
    cmd = torch.CmdLine()
    cmd:option('-epochs', 20, 'Epoches to run')
    cmd:option('-saveEpoch', 0, 'Period to save model during training')
    cmd:option('-saveName', '', 'Filename when saving the model. If not specified, modelType or model will be used')
    return cmd
end

function M.parsedCmdLineToTrainParams(parsed)
    return {
        epochs = parsed.epochs,
        saveEpoch = parsed.saveEpoch,
        saveName = parsed.saveName,
    }
end

function M.main(arg)
    -- arg: command line arguments
    local models = require('models')

    local cmd = util.createModelCmdLine(true)
    cmd = util.mergeCmdLineOptions(cmd, util.createOptimCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, M.createTrainCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, util.createConfigCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, dataset.createDataCmdLine())

    local parsed = cmd:parse(arg or {})
    if parsed.saveName == '' then
        if parsed.model ~= '' then
            local ext = paths.extname(parsed.model)
            local path = paths.dirname(parsed.model)
            local name = paths.basename(parsed.model, ext)
            parsed.saveName = paths.concat(path, name)
        else
            parsed.saveName = parsed.modelType
        end
    end

    local model_params = util.parsedCmdLineToModelParams(parsed)
    local optim_params = util.parsedCmdLineToOptimParams(parsed)
    local train_params = M.parsedCmdLineToTrainParams(parsed)
    local data_params = dataset.parsedCmdLineToDataParams(parsed)
    local config_params = util.parsedCmdLineToConfigParams(parsed)

    if not model_params.model then
        io.stderr:write(cmd:help())
        io.stderr:write('\n')
        return
    end

    local criterion = nn.ClassNLLCriterion()

    if config_params.debug then
        print('=== Training Parameters ===')
        print(train_params)
        print('=== Optimization Parameters ===')
        print(optim_params)
        print('=== Model ===')
        print(model_params.model)
        print(string.format('=== Data: %s ===', train_params.data))
        print(data_params.data)
    end

    -- Cuda-ify
    if config_params.cuda then
        require 'cunn'
        model_params.model:cuda()
        criterion:cuda()
        data_params.data.train.data = data_params.data.train.data:cuda()
        data_params.data.train.labels = data_params.data.train.labels:cuda()
        data_params.data.validate.data = data_params.data.validate.data:cuda()
        data_params.data.validate.labels = data_params.data.validate.labels:cuda()
        data_params.data.test.data = data_params.data.test.data:cuda()
        data_params.data.test.labels = data_params.data.test.labels:cuda()
    end

    M.train(model_params.model, criterion, data_params.data, optim_params, train_params, config_params)
end

return M
