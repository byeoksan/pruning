-- test to compute the number of weights in each layer
models = require 'models'
require 'gnuplot'

model = models.load('allcnn', 10)
print(model)

w, _ = model:parameters()
print(w)
--print(#w)

local w_acc = w[1]:view(w[1]:nElement())
-- w_acc accumulates the number of weights of each layer.
w_layer = torch.Tensor(#w):fill(0)
w_layer[1] = w_acc:nElement()
for i = 2, #w do
	w_layer[i] = w[i]:nElement()
	w_acc = torch.cat(w_acc, w[i]:view(w[i]:nElement()), 1)
end
print(w_layer)

-- w_layer2 contains the number of weights in each layer.
w_layer2 = torch.Tensor(#w/2):fill(0)
for i = 1, #w do
	if i%2 == 0 then
		w_layer2[i/2] = w[i]:nElement() + w[i-1]:nElement()
	end
end

print(w_layer2) 
gnuplot.figure(1)
gnuplot.xlabel('weights')
gnuplot.hist(w_acc, 1000)

