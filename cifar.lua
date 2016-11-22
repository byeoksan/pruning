#!/usr/bin/env th


--[[ CIFAR data directory structure
cifar/
├── cifar-100-t7/
│   ├── data_batch_1.t7
│   ├── data_batch_2.t7
│   ├── data_batch_3.t7
│   ├── data_batch_4.t7
│   ├── data_batch_5.t7
│   ├── meta.t7
│   └── test_batch.t7
└── cifar-10-t7/
    ├── data_batch_1.t7
    ├── data_batch_2.t7
    ├── data_batch_3.t7
    ├── data_batch_4.t7
    ├── data_batch_5.t7
    ├── meta.t7
    └── test_batch.t7
--]]

local M = {}

local function _load_cifar_10()
    local train_data = {
        data = torch.DoubleTensor(50000, 3, 32, 32),
        labels = torch.DoubleTensor(50000)
    }
    local test_data = {
        data = torch.DoubleTensor(10000, 3, 32, 32),
        labels = torch.DoubleTensor(10000)
    }

    -- Load training data
    for i = 1, 5 do
        local lower = 10000*(i-1) + 1
        local upper = 10000*i

        local filename = string.format('cifar/cifar-10-t7/data_batch_%d.t7', i)
        local batch = torch.load(filename)
        train_data.data[{{lower, upper}, {}, {}, {}}] = batch.data:reshape(10000, 3, 32, 32)
        train_data.labels[{{lower, upper}}] = batch.labels + 1
    end

    -- Load test data
    test_batch = torch.load('cifar/cifar-10-t7/test_batch.t7')
    test_data.data[{{}}] = test_batch.data:reshape(10000, 3, 32, 32)
    test_data.labels[{{}}] = test_batch.labels + 1

    mean = torch.mean(train_data.data, 1)
    train_data.data:add(-mean:repeatTensor(50000, 1, 1, 1))
    test_data.data:add(-mean:repeatTensor(10000, 1, 1, 1))

    return {train=train_data, test=test_data}
end

local function _load_cifar_100()
    local train_data = {
        data = torch.DoubleTensor(50000, 3, 32, 32),
        labels = torch.DoubleTensor(50000)
    }
    local test_data = {
        data = torch.DoubleTensor(10000, 3, 32, 32),
        labels = torch.DoubleTensor(10000)
    }

    -- Load training data
    for i = 1, 5 do
        local lower = 10000*(i-1) + 1
        local upper = 10000*i

        local filename = string.format('cifar/cifar-100-t7/data_batch_%d.t7', i)
        local batch = torch.load(filename)
        train_data.data[{{lower, upper}, {}, {}, {}}] = batch.data:reshape(10000, 3, 32, 32)
        train_data.labels[{{lower, upper}}] = batch.fine_labels + 1
    end

    -- Load test data
    test_batch = torch.load('cifar/cifar-100-t7/test_batch.t7')
    test_data.data[{{}}] = test_batch.data:reshape(10000, 3, 32, 32)
    test_data.labels[{{}}] = test_batch.fine_labels + 1

    mean = torch.mean(train_data.data, 1)
    train_data.data:add(-mean:repeatTensor(50000, 1, 1, 1))
    test_data.data:add(-mean:repeatTensor(10000, 1, 1, 1))

    return {train=train_data, test=test_data}
end

function M.load(nclass)
    if nclass == 10 then
        return _load_cifar_10()
    else
        return _load_cifar_100()
    end
end

return M
