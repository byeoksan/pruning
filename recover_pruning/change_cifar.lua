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

local data_st = {
    data = torch.DoubleTensor(10000, 3, 32, 32),
    labels = torch.DoubleTensor(10000)
}

-- Load training data
for i = 1, 5 do
    local filename = string.format('cifar/cifar-10-t7/data_batch_%d.t7', i)
    
    local batch = torch.load(filename)
    
    local target_st = {
        data = torch.DoubleTensor(10000, 3, 32, 32),
        labels = torch.DoubleTensor(10000)
    }

    data_st.data[{{}}] = batch.data
    data_st.labels[{{}}] = batch.labels
    len = 0
    for i = 1, 10000 do
        if data_st.labels[i] < 5 then
            len = len + 1
            target_st.data[len] = data_st.data[i]
            target_st.labels[len] = data_st.labels[i]
        end
    end

    local target = {
        data = torch.ByteTensor(len, 3072),
        batch_label = string.format("training batch %d of 5", i),
        labels = torch.ByteTensor(len, 1)
    }

    target.data[{{}}] = target_st.data:sub(1, len):reshape(len, 3072)
    target.labels[{{}}] = target_st.labels:sub(1, len)

    local target_fname = string.format('cifar/cifar-5-t7/data_batch_%d.t7', i)
    torch.save(target_fname, target)
    print("Save ", target_fname, " done")
end

-- Load test data
test_batch = torch.load('cifar/cifar-10-t7/test_batch.t7')
data_st.data[{{}}] = test_batch.data
data_st.labels[{{}}] = test_batch.labels

local target_st = {
    data = torch.DoubleTensor(10000, 3, 32, 32),
    labels = torch.DoubleTensor(10000)
}

len = 0
for i = 1, 10000 do
    if data_st.labels[i] < 5 then
        len = len + 1
        target_st.data[len] = data_st.data[i]
        target_st.labels[len] = data_st.labels[i]
    end
end

local target = {
    data = torch.ByteTensor(len, 3072),
    batch_label = "testing batch 1 of 1",
    labels = torch.ByteTensor(len, 1)
}

target.data[{{}}] = target_st.data:sub(1, len):reshape(len, 3072)
target.labels[{{}}] = target_st.labels:sub(1, len)

torch.save('cifar/cifar-5-t7/test_batch.t7', target)
print("Save cifar/cifar-5-t7/test_batch.t7 done")


