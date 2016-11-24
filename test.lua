require 'nn'
require 'gnuplot'
require 'SpatialConvolutionWithMask'
require 'LinearWithMask'

modelnoP = torch.load('cnn-mnist-model.t7')
modelP = torch.load('cnn-mnist-model_pruning.t7')

function plotw(w, numFigure)
	local w_acc = w[1]:view(w[1]:nElement())
	for j = 2, #w do
		w_acc = torch.cat(w_acc, w[j]:view(w[j]:nElement()), 1)
	end
	gnuplot.figure(numFigure)
	gnuplot.xlabel('weights')
	gnuplot.hist(w_acc,500)
end

wnoP, _ = modelnoP:parameters()
wP, _ = modelP:getParameters()

plotw(wnoP, 1)
--plotw(wP, 2)
p_num = torch.Tensor(3)
p_num_ori = torch.Tensor(3)

i_n = 1
pruned_cnt = 0
for i, m in ipairs(modelP:findModules('SpatialConvolutionWithMask')) do
	pruned_cnt = pruned_cnt + m.weightMask:eq(0):sum()
	p_num[i_n] = m.weightMask:eq(0):sum()
	i_n = i_n+1
end

for i, m in ipairs(modelP:findModules('LinearWithMask')) do
	pruned_cnt = pruned_cnt + m.weightMask:eq(0):sum()
	p_num[i_n] = m.weightMask:eq(0):sum()
	i_n = i_n+1
end
i_n = 1
for i, m in ipairs(modelnoP:findModules('SpatialConvolutionWithMask')) do
	p_num_ori[i_n] = m.weight:nElement()
	i_n = i_n+1
	print (i_n)
end

for i, m in ipairs(modelnoP:findModules('LinearWithMask')) do
	p_num_ori[i_n] = m.weight:nElement()
	i_n = i_n+1
	print (i_n)
end

print (p_num)
print (p_num_ori)
print (modelP)
print (modelP:parameters())
print (wP:nElement())

wP = wP:reshape(wP:nElement())
_, indices = torch.abs(wP):sort()
indices = indices[{{pruned_cnt+1, -1}}]

lal = torch.ByteTensor(wP:size(1)):fill(0)
for i = 1, indices:size(1) do
	lal[indices[i]] = 1
end

gnuplot.figure()
gnuplot.xlabel('weights')
gnuplot.hist(wP[lal], 500)
