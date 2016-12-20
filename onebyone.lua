#!/usr/bin/env th

local test = require('test')
local dataset = require('dataset')
local models = require('models')
local train = require('train')
local util = require('util')

local M = {}

function M.one_by_one_pruning(QList, mult, step, model, criterion, data, opt_params, config_params)
	layers = models.get_prunables(model)
	if #layers ~= #QList then
		io.stderr:write(string.format('Error! number of prunable layers(%d) not matches number of QFactors(%d)\n', #layers, #QList))
		return
	end
	for i = 1, #layers do
		local target = layers[i]
		local QF = QList[i]
		print('======================================================')
		print(string.format('### Pruning layers[%d] => %s', i, target))
		for r = 1, step do
			print('------------------------------------------------------')
			print(string.format('\tQFactor: %f', QF))
			target:pruneQfactor(QF)
			train.train(model, criterion, data, opt_params, config_params)
			QF = QF * mult
		end
	end
end

function M.main(arg)
    -- arg: command line arguments
    local models = require('models')

    local cmd = torch.CmdLine()
    cmd:option('-multiplier', '1', 'Multiplier...')
    cmd:option('-qFactor', '', 'QFactor...')
    cmd:option('-step', '1', 'step...')
    cmd:option('-model', '', 'Model to resume')
    cmd:option('-learningRate', 0.01, 'Initial learning rate')
    cmd:option('-learningRateDecay', 1e-4, 'Learning rate decay')
    cmd:option('-weightDecay', 0.0005, 'Weight decay')
    cmd:option('-momentum', 0.9, 'Learning momentum')
    cmd:option('-cuda', false, 'Whether to use cuda')
    cmd:option('-batch', 128, 'Batch size')
    cmd:option('-epochs', 20, 'Epoches to run')
    cmd:option('-saveEpoch', 0, 'Period to save model during training')
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
    else
        io.stderr:write(cmd:help())
        io.stderr:write('\n')
        return
    end
	
	if params.qFactor == '' then
		io.stderr:write('Empty qFactor list...\n')
		return
	else
		QList = util.csvToList(params.qFactor)
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

    M.one_by_one_pruning(QList, params.multiplier, params.step, model, criterion, data, opt_params, config_params)
end

return M
