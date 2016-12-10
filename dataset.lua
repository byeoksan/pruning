#!/usr/bin/env th

local M = {}

function M.load()
    local mnist = require('mnist')
    local data = {
        train = {},
        validate = {},
        test = {},
    }

    local d = mnist.traindataset()
    local shape = d.data:size()

    shape[1] = 50000
    data.train.data = torch.DoubleTensor(shape)
    data.train.labels = torch.DoubleTensor(50000)
    data.train.data:copy(d.data[{{1, 50000}}])
    data.train.labels:copy(d.label[{{1, 50000}}])

    shape[1] = 10000
    data.validate.data = torch.DoubleTensor(shape)
    data.validate.labels = torch.DoubleTensor(10000)
    data.validate.data:copy(d.data[{{50001, 60000}}])
    data.validate.labels:copy(d.label[{{50001, 60000}}])

    d = mnist.testdataset()
    shape = d.data:size()

    data.test.data = torch.DoubleTensor(shape)
    data.test.labels = torch.DoubleTensor(shape[1])
    data.test.data:copy(d.data)
    data.test.labels:copy(d.label)

    -- Normalize
    local mean = data.train.data:mean()
    local std = data.train.data:std()
    data.train.data:add(-mean)
    data.validate.data:add(-mean)
    data.test.data:add(-mean)

    data.train.data:div(std)
    data.validate.data:div(std)
    data.test.data:div(std)

    -- Label adjust
    data.train.labels:add(1)
    data.validate.labels:add(1)
    data.test.labels:add(1)

    return data
end

return M
