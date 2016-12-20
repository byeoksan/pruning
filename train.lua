#!/usr/bin/env th

require 'paths'
require 'nn'
require 'optim'

local util = require('util')
local progress = require('progress')
local dataset = require('dataset')
local test = require('test')

local M = {}

local function _step(model, criterion, data, opt_params, config_params)
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

        _, fs = optim.sgd(eval, weight, opt_params)
        loss = loss + fs[1]

        count = count + 1
    end
    progress.clear()

    return loss / count
end

function M.train(model, criterion, data, opt_params, config_params)
    local saveEpoch = config_params.saveEpoch or 0
    local saveName = config_params.saveName
    local epochs = config_params.epochs or 10

    for epoch = 1, epochs do
        print(string.format('Training Epoch %d...', epoch))
        local loss = _step(model, criterion, data, opt_params, config_params)
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
    cmd:option('-modelType', '', 'Model to learn (if model is specified, this is ignored')
    cmd:option('-model', '', 'Model to resume')
    cmd:option('-cuda', false, 'Whether to use cuda')
    cmd:option('-batch', 128, 'Batch size')
    cmd:option('-epochs', 20, 'Epoches to run')
    cmd:option('-saveEpoch', 0, 'Period to save model during training')
    cmd:option('-progress', false, 'True to show progress')
    cmd:option('-saveName', '', 'Filename when saving the model. If not specified, modelType or model will be used')
    cmd:option('-debug', false, 'True for debugging')
    return cmd
end

function M.parsedCmdLineToTrainParams(parsed)
    return {
        cuda = parsed.cuda,
        batch = parsed.batch,
        epochs = parsed.epochs,
        saveEpoch = parsed.saveEpoch,
        saveName = parsed.saveName,
        progress = parsed.progress,
        debug = parsed.debug,
    }
end

function M.main(arg)
    -- arg: command line arguments
    local models = require('models')

    local cmd = torch.CmdLine()
    cmd:option('-modelType', '', 'Model to learn (if model is specified, this is ignored')
    cmd:option('-model', '', 'Model to resume')

    cmd = util.mergeCmdLineOptions(cmd, util.createOptimCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, M.createTrainCmdLine())

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

    opt_params = util.parsedCmdLineToOptimParams(params)
    config_params = M.parsedCmdLineToTrainParams(params)

    if config_params.debug then
        print('=== Training Parameters ===')
        print(config_params)
        print('=== Optimization Parameters ===')
        print(opt_params)
        print('=== Model ===')
        print(model)
    end

    -- Load the data
    local data = dataset.load()
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
        data.validate.data = data.validate.data:cuda()
        data.validate.labels = data.validate.labels:cuda()
        data.test.data = data.test.data:cuda()
        data.test.labels = data.test.labels:cuda()
    end

    M.train(model, criterion, data, opt_params, config_params)
end

return M
