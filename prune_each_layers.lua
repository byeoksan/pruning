#!/usr/bin/env th

local test = require('test')
local dataset = require('dataset')
local models = require('models')

local M = {}

function M.prune_each_layers(model)
	data = dataset.load()
	layers = models.get_prunables(model)
	for i = 1, #layers do
		local target = layers[i]
		target:stash()
		print('======================================================')
		print(string.format('### Pruning layers[%d] => %s', i, target))
		for rate = 0, 0.9, 0.1 do
			print('------------------------------------------------------')
			print(string.format('\tPruning Ratio = %f', rate))
			target:pruneRatio(rate)
			result = test.evaluate(model, data.test.data, data.test.labels, {batch=128})
			print(string.format('\tEvaluate accuracy = %f', result))
			target:stashPop()
		end
	end
end

return M
