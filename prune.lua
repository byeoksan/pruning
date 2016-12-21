#!/usr/bin/env th

local models = require('models')
local train = require('train')
local test = require('test')
local dataset = require('dataset')
local util = require('util')

local M = {}

local prune_methods = OrderedMap({
    {qfactor = 'pruneQfactor'},
    {ratio = 'pruneRatio'},
    {sensitivity = 'pruneSensitivity'},
})

local function _pruneIndividual(model, prune_params, data_params, optim_params, train_params, config_params)
    local init = prune_params.init
    local mult = prune_params.mult
    local step = prune_params.stepIter
    local method = prune_params.method

    local current_factor = init:clone()
    local layers = models.get_prunables(model)

    for index, layer in ipairs(layers) do
        for i = 1, step do
            local factor = current_factor[index]
            print(string.format('Iteration %d (%s, factor:%s)', i, layer, factor))
            layer[prune_methods[method]](layer, factor)
            current_factor[index] = factor * mult[index]
            if not prune_params.notrain then
                train.train(model, data_params.data, optim_params, train_params, config_params)
            else
                local acc_train = test.evaluate(model, data_params.data.train.data, data_params.data.train.labels, config_params)
                print(string.format('\tTrain Accuracy: %f', acc_train*100))
                local acc_validate = test.evaluate(model, data_params.data.validate.data, data_params.data.validate.labels, config_params)
                print(string.format('\tValidate Accuracy: %f', acc_validate*100))
                local acc_test = test.evaluate(model, data_params.data.test.data, data_params.data.test.labels, config_params)
                print(string.format('\tTest Accuracy: %f', acc_test*100))
            end
        end
    end
end

local function _pruneType(model, prune_params, data_params, optim_params, train_params, config_params)
    local init = prune_params.init
    local mult = prune_params.mult
    local step = prune_params.stepIter
    local method = prune_params.method

    local current_factor = init:clone()
    local layer_map = models.get_prunables_by_type(model)

    for layer_type in layer_map:keys():iterate() do
        local layers_and_indices = layer_map[layer_type]

        for i = 1, step do
            -- Prune
            for layer_and_index in layers_and_indices:iterate() do
                local layer = layer_and_index[1]
                local index = layer_and_index[2]

                local factor = current_factor[index]
                -- TODO: take method from the user
                layer[prune_methods[method]](layer, factor)
                current_factor[index] = factor * mult[index]
            end

            print(string.format('Iteration %d (%s)', i, layer_type))
            if not prune_params.notrain then
                train.train(model, data_params.data, optim_params, train_params, config_params)
            else
                local acc_train = test.evaluate(model, data_params.data.train.data, data_params.data.train.labels, config_params)
                print(string.format('\tTrain Accuracy: %f', acc_train*100))
                local acc_validate = test.evaluate(model, data_params.data.validate.data, data_params.data.validate.labels, config_params)
                print(string.format('\tValidate Accuracy: %f', acc_validate*100))
                local acc_test = test.evaluate(model, data_params.data.test.data, data_params.data.test.labels, config_params)
                print(string.format('\tTest Accuracy: %f', acc_test*100))
            end
        end
    end
end

local function _pruneAll(model, prune_params, data_params, optim_params, train_params, config_params)
    local init = prune_params.init
    local mult = prune_params.mult
    local step = prune_params.stepIter
    local method = prune_params.method

    local current_factor = init:clone()
    local layers = models.get_prunables(model)

    for i = 1, step do
        for index, layer in ipairs(layers) do
            local factor = current_factor[index]
            layer[prune_methods[method]](layer, factor)
            current_factor[index] = factor * mult[index]
        end

        print(string.format('Iteration %d (ALL)'))
        if not prune_params.notrain then
            train.train(model, data_params.data, optim_params, train_params, config_params)
        else
            local acc_train = test.evaluate(model, data_params.data.train.data, data_params.data.train.labels, config_params)
            print(string.format('\tTrain Accuracy: %f', acc_train*100))
            local acc_validate = test.evaluate(model, data_params.data.validate.data, data_params.data.validate.labels, config_params)
            print(string.format('\tValidate Accuracy: %f', acc_validate*100))
            local acc_test = test.evaluate(model, data_params.data.test.data, data_params.data.test.labels, config_params)
            print(string.format('\tTest Accuracy: %f', acc_test*100))
        end
    end
end

local prune_function_by_group = OrderedMap({
    {individual = _pruneIndividual},
    {type = _pruneType},
    {all = _pruneAll},
})

function M.createPruneCmdLine()
    cmd = torch.CmdLine()
    cmd:option('-method', prune_methods:keys()[1], string.format('Method to prune (%s)', tostring(prune_methods:keys()):sub(2, -2):gsub(',', ', ')))
    cmd:option('-group', prune_function_by_group:keys()[1], string.format('Group to apply pruning (%s)', tostring(prune_function_by_group:keys()):sub(2, -2):gsub(',', ', ')))
    cmd:option('-init', '', 'Initial factors to be applied in pruning (comma-separated)')
    cmd:option('-mult', '', 'Multiplier of factors (comma-separated, same size as init)')
    cmd:option('-stepIter', 1, 'Number of pruning steps to prune next layer(s)')
    cmd:option('-notrain', false, 'True if training is not needed')
    return cmd
end

function M.parsedCmdLineToPruneParams(parsed)
    return {
        method = parsed.method,
        group = parsed.group,
        init = util.csvToList(parsed.init),
        mult = util.csvToList(parsed.mult),
        stepIter = parsed.stepIter,
        notrain = parsed.notrain,
    }
end

function M.main(arg)
    local cmd = M.createPruneCmdLine()
    cmd = util.mergeCmdLineOptions(cmd, dataset.createDataCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, util.createConfigCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, train.createTrainCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, util.createOptimCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, util.createModelCmdLine(false))

    local parsed = cmd:parse(arg)
    train.updateSaveName(parsed)

    local prune_params = M.parsedCmdLineToPruneParams(parsed)
    local config_params = util.parsedCmdLineToConfigParams(parsed)
    local model_params = util.parsedCmdLineToModelParams(parsed)
    local data_params = dataset.parsedCmdLineToDataParams(parsed)
    local train_params = train.parsedCmdLineToTrainParams(parsed)
    local optim_params = util.parsedCmdLineToOptimParams(parsed)

    if not model_params.model then
        io.stderr:write('Model should be specified\n')
        return
    end

    local layers = models.get_prunables(model_params.model)
    if #layers ~= #prune_params.init then
        io.stderr:write(string.format('The number of prunable layers (%d) does not match to init (%s)\n', #layers, prune_params.init))
        return
    end

    if #layers ~= #prune_params.mult then
        io.stderr:write(string.format('The number of prunable layers (%d) does not match to mult (%s)\n', #layers, prune_params.mult))
        return
    end

    -- Cuda-ify
    if config_params.cuda then
        require 'cunn'
        model_params.model:cuda()
        data_params.data.train.data = data_params.data.train.data:cuda()
        data_params.data.train.labels = data_params.data.train.labels:cuda()
        data_params.data.validate.data = data_params.data.validate.data:cuda()
        data_params.data.validate.labels = data_params.data.validate.labels:cuda()
        data_params.data.test.data = data_params.data.test.data:cuda()
        data_params.data.test.labels = data_params.data.test.labels:cuda()
    end

    prune_function_by_group[prune_params.group](model_params.model, prune_params, data_params, optim_params, train_params, config_params)
end

return M
