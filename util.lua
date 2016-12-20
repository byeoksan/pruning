#!/usr/bin/env th

local models = require('models')

local M = {}

function M.csvToList(csv)
    if torch.type(csv) ~= 'string' then
        return List{}
    end

    csv = string.gsub(csv, ' ', '')
    list = List(string.split(csv, ',')):map(tonumber)

    if list:contains(false) then
        return List{}
    end

    return list
end

function M.mergeCmdLineOptions(cmd1, cmd2)
    cmd = torch.CmdLine()
    for key, option in pairs(cmd1.options) do
        if cmd.options[key] then
            cmd.options[key].default = option.default
            cmd.options[key].help = option.help
            cmd.options[key]._type_ = option._type_
        else
            cmd:option(option.key, option.default, option.help, option._type_)
        end
    end

    for key, option in pairs(cmd2.options) do
        if cmd.options[key] then
            cmd.options[key].default = option.default
            cmd.options[key].help = option.help
            cmd.options[key]._type_ = option._type_
        else
            cmd:option(option.key, option.default, option.help, option._type_)
        end
    end

    return cmd
end

function M.createOptimCmdLine()
    cmd = torch.CmdLine()
    cmd:option('-learningRate', 0.01, 'Initial learning rate')
    cmd:option('-learningRateDecay', 1e-4, 'Learning rate decay')
    cmd:option('-weightDecay', 0.0005, 'Weight decay')
    cmd:option('-momentum', 0.9, 'Learning momentum')

    return cmd
end

function M.parsedCmdLineToOptimParams(parsed)
    return {
        learningRate = parsed.learningRate,
        learningRateDecay = parsed.learningRateDecay,
        weightDecay = parsed.weightDecay,
        momentum = parsed.momentum,
    }
end

function M.createModelCmdLine(useType)
    cmd = torch.CmdLine()
    cmd:option('-model', '', 'Model to resume')
    if useType then
        cmd:option('-modelType', '', 'Model type to create (if model is specified, this is ignored')
        cmd:option('-nclass', 0, 'Number of classes to classify')
    end
    return cmd
end

function M.parsedCmdLineToModelParams(parsed)
    local model
    if parsed.model and parsed.model ~= '' then
        model = models.restore(parsed.model)
    elseif parsed.modelType and parsed.modelType ~= '' then
        model = models.load(parsed.modelType, parsed.nclass)
    end

    return {model = model, nclass = parsed.nclass}
end

function M.createConfigCmdLine()
    cmd = torch.CmdLine()
    cmd:option('-cuda', false, 'Whether to use cuda')
    cmd:option('-batch', 128, 'Batch size')
    cmd:option('-progress', false, 'True to show progress')
    cmd:option('-debug', false, 'True for debugging')
    return cmd
end

function M.parsedCmdLineToConfigParams(parsed)
    return {
        cuda = parsed.cuda,
        batch = parsed.batch,
        progress = parsed.progress,
        debug = parsed.debug,
    }
end

return M
