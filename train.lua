#!/usr/bin/env th

require 'paths'
require 'nn'
require 'optim'
require 'xlua'
require 'cifar'

test = require('test')

local M = {}

-- TODO: Use MNIST
local function _prepare_data()
    local cifar = require('cifar')
    return cifar.load(5)
end

local function _step(model, criterion, data, opt_params, config_params)
    model:training()
    local batch = config_params.batch
    local shape = data.train.data:size()
    local data_size = shape[1]
    local perm = torch.randperm(data_size):long()
    local loss = 0

    local count = 0
    for b = 1, data_size, batch do
        local mini_batch = math.min(b+batch-1, data_size) - b + 1
        if config_params.progress then
            xlua.progress(b, data_size)
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

        _, fs = optim.sgd(eval, weight, opt_params)
        loss = loss + fs[1]

        count = count + 1
    end

    return loss / count
end

local function _train(model, criterion, data, opt_params, config_params)
    local saveEpoch = config_params.saveEpoch
    local saveName = config_params.saveName

    for epoch = 1, config_params.epochs do
        print(string.format('Training Epoch %d...', epoch))
        local loss = _step(model, criterion, data, opt_params, config_params)
        print(string.format('\tTrain loss: %f', loss))

        -- Intermmediate Save
        if (saveEpoch > 0 and (epoch % saveEpoch) == 0) or epoch == config_params.epochs then
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
        local test_accuracy = test.evaluate(model, data.test.data, data.test.labels, config_params)
        print(string.format('\tTest Accuracy: %f', test_accuracy * 100))
    end
end

function M.main(arg)
    -- arg: command line arguments
    local models = require('models')

    local cmd = torch.CmdLine()
    cmd:option('-modelType', '', 'Model to learn (if model is specified, this is ignored')
    cmd:option('-model', '', 'Model to resume')
    cmd:option('-learningRate', 0.01, 'Initial learning rate')
    cmd:option('-learningRateDecay', 1e-4, 'Learning rate decay')
    cmd:option('-weightDecay', 0.0005, 'Weight decay')
    cmd:option('-momentum', 0.9, 'Learning momentum')
    cmd:option('-cuda', false, 'Whether to use cuda')
    cmd:option('-batch', 128, 'Batch size')
    cmd:option('-epochs', 20, 'Epoches to run')
    cmd:option('-saveEpoch', 0, 'Period to save model during training')
    cmd:option('-nclass', 10, 'Number of classes of CIFAR (5, 10, 100)')
    cmd:option('-progress', false, 'True to show progress')
    cmd:option('-saveName', '', 'Filename when saving the model. If not specified, modelType or model will be used')
    cmd:option('-debug', false, 'True for debugging')

    local params = cmd:parse(arg or {})

    -- Load the model
    local model
    if params.model ~= '' then
        -- TODO: Sanity check and implement "restore"
        model = models.restore(params.model)

        local ext = paths.extname(params.model)
        local path = paths.dirname(params.model)
        local name = paths.basename(params.model, ext)
        if params.saveName == '' then
            params.saveName = paths.concat(path, name)
        end

    elseif params.modelType ~= '' then
        -- Sanity check
        model = models.load(params.modelType, 5)

        if params.saveName == '' then
            params.saveName = params.modelType
        end
    else
        io.stderr:write(cmd:help())
        io.stderr:write('\n')
        return
    end

    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()

    opt_params = {
        learningRate = params.learningRate,
        learningRateDecay = params.learningRateDecay,
        weightDecay = params.weightDecay,
        momentum = params.momentum,
    }

    config_params = {
        cuda = params.cuda,
        batch = params.batch,
        epochs = params.epochs,
        saveEpoch = params.saveEpoch,
        saveName = params.saveName,
        nclass = params.nclass,
        progress = params.progress,
        debug = params.debug,
    }

    if config_params.debug then
        print('=== Training Parameters ===')
        print(config_params)
        print('=== Optimization Parameters ===')
        print(opt_params)
        print('=== Model ===')
        print(model)
    end

    -- Load the data
    -- TODO: Use MNIST
    local data = _prepare_data()
    if config_params.debug then
        print(data)
    end

    -- Cuda-ify
    if params.cuda then
        require 'cunn'
        model:cuda()
        criterion:cuda()
        data.train.data = data.train.data:cuda()
        data.train.labels = data.train.labels:cuda()
        data.test.data = data.test.data:cuda()
        data.test.labels = data.test.labels:cuda()
    end

    _train(model, criterion, data, opt_params, config_params)
end

return M
