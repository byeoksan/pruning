#!/usr/bin/env th

local util = require('util')
local models = require('models')
local dataset = require('dataset')
local test = require('test')

local M = {}

local function _probe(model, data, probe_params, config_params)
    local result = List()
    local layers = models.get_prunables(model)

    for i, layer in ipairs(layers) do
        local sub_result = List()

        if config_params.debug then
            print(string.format('Layer %d [%s]', i, layer))
        end

        layer:stash()
        for ratio = 0, 1, probe_params.interval do
            layer:pruneRatio(ratio)

            local acc = Map()
            acc.ratio = ratio
            acc.train = test.evaluate(model, data.train.data, data.train.labels, config_params)
            acc.validate = test.evaluate(model, data.validate.data, data.validate.labels, config_params)
            acc.test = test.evaluate(model, data.test.data, data.test.labels, config_params)

            if config_params.debug then
                print(string.format('\t%.2f%%: %.2f%% / %.2f%% / %.2f%%', ratio, acc.train*100, acc.validate*100, acc.test*100))
            end

            sub_result:append(acc)
            layer:stashPop()
        end

        result:append(sub_result)
    end

    torch.save(probe_params.saveName, result)
end

function M.createProbeCmdLine()
    local cmd = torch.CmdLine()
    cmd:option('-interval', '0.05', 'Ratio interval')
    cmd:option('-saveName', 'probe_result.t7', 'File name to store the result')
    return cmd
end

function M.parsedCmdLineToProbeParams(parsed)
    return {
        interval = parsed.interval,
        saveName = parsed.saveName,
    }
end

function M.main(arg)
    local cmd = util.createModelCmdLine(false)
    cmd = util.mergeCmdLineOptions(cmd, M.createProbeCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, dataset.createDataCmdLine())
    cmd = util.mergeCmdLineOptions(cmd, util.createConfigCmdLine())

    local parsed = cmd:parse(arg)

    local probe_params = M.parsedCmdLineToProbeParams(parsed)
    local model_params = util.parsedCmdLineToModelParams(parsed)
    local data_params = dataset.parsedCmdLineToDataParams(parsed)
    local config_params = util.parsedCmdLineToConfigParams(parsed)

    if not model_params.model then
        io.stderr:write('Model should be given\n')
        return
    end

    _probe(model_params.model, data_params.data, probe_params, config_params)
end

return M
