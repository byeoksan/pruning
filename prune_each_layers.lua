#!/usr/bin/env th

local test = require('test')
local dataset = require('dataset')
local models = require('models')

local M = {}

local function wprint(f, msg)
	print(msg)
	f:write(msg)
	f:write('\n')
end

function M.prune_each_layers(model, fName)
	f = io.open(fName, 'w')
	data = dataset.load()
	layers = models.get_prunables(model)
	for i = 1, #layers do
		local target = layers[i]
		target:stash()
		wprint(f, '======================================================')
		wprint(f, string.format('### Pruning layers[%d] => %s', i, target))
		for rate = 0, 0.9, 0.1 do
			wprint(f, '------------------------------------------------------')
			wprint(f, string.format('\tPruning Ratio: %f', rate))
			target:pruneRatio(rate)
			
			train_accuracy = test.evaluate(model, data.train.data, data.train.labels, {batch=128})
    		wprint(f, string.format('\tTrain Accuracy: %f', train_accuracy * 100))
			validate_accuracy = test.evaluate(model, data.validate.data, data.validate.labels, {batch=128})
    		wprint(f, string.format('\tValidate Accuracy: %f', validate_accuracy * 100))
			test_accuracy = test.evaluate(model, data.test.data, data.test.labels, {batch=128})
    		wprint(f, string.format('\tTest Accuracy: %f', test_accuracy * 100))
			target:stashPop()
		end
	end
	io.close(f)
end

function M.main(arg)
	local cmd = torch.CmdLine()
    cmd:option('-model', '', 'Model to resume')
    cmd:option('-saveName', '', 'Filename when saving the model. If not specified, modelType or model will be used')
    
	local params = cmd:parse(arg or {})
	if (params.model == '' or params.saveName == '') then
		io.stderr:write(cmd:help())
		io.stderr:write('\n')
		return
	end

	local model = models.restore(params.model)
	local fName = string.format('%s.log', params.saveName)
	M.prune_each_layers(model, fName)
end

M.main(arg)

return M
