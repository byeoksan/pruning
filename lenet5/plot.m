% Plot Accuracy curves
clear
close all
clc

learningrate = 0.1;
model = 'lenet';
filenameW = strcat(model, '_r_', num2str(learningrate), '_w.mat');
filenameAcc = strcat(model, '_r_', num2str(learningrate), '.mat');
ojbW = load(filenameW);
objAcc = load(filenameAcc);

fprintf('Testset Accuracy: %0.2f\n', objAcc.acc_test)

% Weight Histograms: Convolution layers
binConv = 50;
figure(1)
subplot(2,1,1)
hist(ojbW.w_conv1, binConv)
xlabel('weight')
title('Convolution Layer 1')

subplot(2,1,2)
hist(ojbW.w_conv2, binConv)
xlabel('weight')
title('Convolution Layer 2')

% Weight Histograms: Convolution layers
binFc = 100;
figure(2)
subplot(3,1,1)
hist(ojbW.w_fc1, binFc)
xlabel('weight')
title('Fully-connected Layer 1')

subplot(3,1,2)
hist(ojbW.w_fc2, binFc)
xlabel('weight')
title('Fully-connected Layer 2')

subplot(3,1,3)
hist(ojbW.w_fc3, binFc)
xlabel('weight')
title('Fully-connected Layer 3')