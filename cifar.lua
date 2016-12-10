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
│   ├── data_batch_1.t7
│   ├── data_batch_2.t7
│   ├── data_batch_3.t7
│   ├── data_batch_4.t7
│   ├── data_batch_5.t7
│   ├── meta.t7
│   └── test_batch.t7
└── cifar-5-t7/
    ├── data_batch_1.t7
    ├── data_batch_2.t7
    ├── data_batch_3.t7
    ├── data_batch_4.t7
    ├── data_batch_5.t7
    ├── meta.t7
    └── test_batch.t7
--]]

local M = {}

function M.load(nclass)
    -- Maximum 50000 data
    local train_data = {
        data = torch.DoubleTensor(50000, 3, 32, 32),
        labels = torch.DoubleTensor(50000)
    }

    -- Maximum 10000 data
    local test_data = {
        data = torch.DoubleTensor(10000, 3, 32, 32),
        labels = torch.DoubleTensor(10000)
    }

    local data = 'data'
    local labels
    if nclass == 100 then
        labels = 'fine_labels'
    else
        labels = 'labels'
    end

    local train_size = 0
    for i = 1, 5 do
        local filename = string.format('cifar/cifar-%d-t7/data_batch_%d.t7', nclass, i)
        local batch = torch.load(filename)
        local batch_size = batch[data]:size(1)

        train_data.data[{{train_size+1, train_size+batch_size}}] = batch[data]:reshape(batch_size, 3, 32, 32)
        train_data.labels[{{train_size+1, train_size+batch_size}}] = batch[labels] + 1

        train_size = train_size + batch_size
    end

    -- Load test data
    local test_batch = torch.load(string.format('cifar/cifar-%d-t7/test_batch.t7', nclass))
    local test_size = test_batch[data]:size(1)
    test_data.data[{{1, test_size}}] = test_batch[data]:reshape(test_size, 3, 32, 32)
    test_data.labels[{{1, test_size}}] = test_batch[labels] + 1

    train_data.data = train_data.data[{{1, train_size}}]
    train_data.labels = train_data.labels[{{1, train_size}}]
    test_data.data = test_data.data[{{1, test_size}}]
    test_data.labels = test_data.labels[{{1, test_size}}]

    -- Normalize
    mean = torch.mean(train_data.data, 1):squeeze()
    for i = 1, train_size do
        train_data.data[i]:csub(mean)
    end
    for i = 1, test_size do
        test_data.data[i]:csub(mean)
    end

    return {train=train_data, test=test_data}
end

return M
