-------------------------------------------------------------------------------------
-- Reference: github.com/szagoruyko/cirfar.torch/provider.lua
--
--	+ First, call M.init(), which returns tables of trainData, testData
--	+ Each table contains tensors of data and labels, and a function.
-- 	+ trainData = {data, labels, size=function() return trsize end}
--	+ testData = {data, labels, size=function() return tesize end}
--
--	+ Secondly, call M.normalize(trainData,testData) once you called M.init().
--	+ The tables you get from M.init() will be normalized for learning.
--------------------------------------------------------------------------------------

require 'nn'
require 'image'
require 'xlua'
require 'torch'

local M = {}

function M.init()
	local trsize = 50000
	local tesize = 10000

	if not paths.dirp('cifar-10-batches-t7') then
		local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
		local tar = paths.basename(www) -- returns the last path component of path
		os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
	end
	
	local trainData = {
		data = torch.Tensor(trsize, 3*32*32),
		labels = torch.Tensor(trsize),
		--data = torch.ByteTensor(trsize, 3*32*32),
		--labels = torch.ByteTensor(trsize),
		size = function() return trsize end
	}
	
	for i = 0, 4 do
		local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
		trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
		trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
	end
	trainData.labels = trainData.labels + 1


	local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
	local testData = {
		data = subset.data:t():double(),
		labels = subset.labels[1]:double(),
		size = function() return tesize end
	}
	--local testData = self.testData
	testData.labels = testData.labels + 1

	-- You can resize dataset if you want by setting diff. values in trsize, tesize
	print(trainData.data:size())
--	trainData.data = trainData.data[{ {1,trsisze} }]
--	trainData.labels = trainData.labels[{ {1,trsize} }]
--	testData.data = testData.data[{ {1, tesize} }]
--	testData.labels = testData.labels[{ {1, tesize} }]

	-- Reshape data tensor into (3x32x32) tensor
	print(trainData.data:size())
	trainData.data = trainData.data:reshape(trsize,3,32,32)
	testData.data = testData.data:reshape(tesize,3,32,32)

	return trainData, testData
end
	

-- preprocess/normalize train/test data
function M.normalize(trainData, testData)

	print '<trainer> preprocessing data (color space + normalization)'
	collectgarbage()

	-- preprocess trainset using LCN(Local Contrast Normalization)
	local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
	for i = 1, trainData:size() do
		xlua.progress(i, trainData:size()) -- displays a progress bar
		-- rgb -> yuv
		local rgb = trainData.data[i]
		local yuv = image.rgb2yuv(rgb)
		-- normalize y locally
		yuv[1] = normalization(yuv[{{1}}])
		trainData.data[i] = yuv
	end
	
	-- normalize u globally
	local mean_u = trainData.data:select(2,2):mean()
	local std_u = trainData.data:select(2,2):std()
	trainData.data:select(2,2):add(-mean_u)
	trainData.data:select(2,2):div(std_u)
	-- normalize v globally
	local mean_v = trainData.data:select(2,3):mean()
	local std_v = trainData.data:select(2,3):std()
	trainData.data:select(2,3):add(-mean_v)
	trainData.data:select(2,3):div(std_v)
	
	-- store the mean and std values before normalization
	trainData.mean_u = mean_u
	trainData.std_u = std_u
	trainData.mean_v = mean_v
	trainData.std_v = std_v

	-- preprocess testset
	for i = 1, testData:size() do
		xlua.progress(i, testData:size())
		-- rgb -> yuv
		local rgb = testData.data[i]
		local yuv = image.rgb2yuv(rgb)
		-- normalize y locaaly
		yuv[{1}] = normalization(yuv[{{1}}])
		testData.data[i] = yuv
	end

	-- normalize u globally, normlize with mean and std from trainset!?
	testData.data:select(2,2):add(-mean_u)
	testData.data:select(2,2):div(std_u)
	-- normalize v globally
	testData.data:select(2,3):add(-mean_v)
	testData.data:select(2,3):add(std_v)
end

return M
