% Plot Accuracy curves
clear
close all
clc

%% Load *.mat files
model = 'lenet';
fileWeightPretrain = strcat(model, 'WeightPretrain.mat');
fileWeightRetrain = strcat(model, 'WeightRetrained.mat');
fileNumWeights = strcat(model, 'NumWeight.mat');
fileAccLoss = strcat(model, 'AccLoss.mat');
WeightPretrain = load(fileWeightPretrain);
WeightRetrain = load(fileWeightRetrain);
WeightNum = load(fileNumWeights);
AccLoss = load(fileAccLoss);
%% Pretraining plots
% Accuracy Plot
fprintf('* Pretraining Plots\n\n')
figure(1)
subplot(2,1,1)
plot(AccLoss.LossPretrain)
xlabel('epoch')
ylabel('Loss')
title('Loss on Pretraining phase')

subplot(2,1,2)
plot(0:size(AccLoss.AccTrainPretrain,1)-1, AccLoss.AccTrainPretrain, 'r', ...
        0:size(AccLoss.AccTrainPretrain,1)-1, AccLoss.AccValidPretrain, 'b')
xlabel('epoch')
ylabel('Accuracy(%)')
legend('Train Set', 'Validation Set')
title('Accuracy on Pretraining phase')


numPreConv1 = size(WeightPretrain.Conv1,1);
numPreConv2 = size(WeightPretrain.Conv2,1);
numPreFc1 = size(WeightPretrain.Fc1,1);
numPreFc2 = size(WeightPretrain.Fc2,1);
numPreFc3 = size(WeightPretrain.Fc3,1);
numPreTotal = numPreConv1 + numPreConv2 + ...
                numPreFc1 + numPreFc2 + numPreFc3;
% Weight Histograms: Convolution layers
binConv = [20, 20];
figure(2)
subplot(3,2,1)
hist(WeightPretrain.Conv1, binConv(1))
xlabel('weight')
title('Convolution Layer 1')
fprintf('  # of weights in Conv1 = %d\n', numPreConv1)

subplot(3,2,2)
hist(WeightPretrain.Conv2, binConv(2))
xlabel('weight')
title('Convolution Layer 2')
fprintf('  # of weights in Conv2 = %d\n', numPreConv2)

% Weight Histograms: Convolution layers
binFc = [100, 50, 25];
subplot(3,2,3)
hist(WeightPretrain.Fc1, binFc(1))
xlabel('weight')
title('Fully-connected Layer 1')
fprintf('  # of weights in Fc1 = %d\n', numPreFc1)

subplot(3,2,4)
hist(WeightPretrain.Fc2, binFc(2))
xlabel('weight')
title('Fully-connected Layer 2')
fprintf('  # of weights in Fc2 = %d\n', numPreFc2)

subplot(3,2,5)
hist(WeightPretrain.Fc3, binFc(3))
xlabel('weight')
title('Fully-connected Layer 3')
fprintf('  # of weights in Fc3 = %d\n', numPreFc3)
fprintf('  Pretraining Testset Accuracy: %0.3f\n', AccLoss.AccTestPretrain)

%% Retraining plots
fprintf('\n* Retraining Plots\n\n')
numReConv1 = size(WeightRetrain.Conv1,1);
numReConv2 = size(WeightRetrain.Conv2,1);
numReFc1 = size(WeightRetrain.Fc1,1);
numReFc2 = size(WeightRetrain.Fc2,1);
numReFc3 = size(WeightRetrain.Fc3,1);
numReTotal = numReConv1 + numReConv2 + ...
                numReFc1 + numReFc2 + numReFc3;

% num of weights over iterative pruning phase
figure(3)
numW = ones(size(WeightNum.numWeight,1)+1,1)*numPreTotal;
numW(2:size(WeightNum.numWeight,1)+1) = WeightNum.numWeight;
plot(0:(size(numW,1)-1), ones(size(numW,1),1)*numPreTotal, '--r', ...
        (0:size(numW,1)-1), numW, 'b')
xlabel('# of pruning')
ylabel('# of weights')
legend('Before Pruning', 'After Pruning')


% Weight Histograms: Convolution layers
binConv = [20, 20];
figure(4)
subplot(3,2,1)
hist(WeightRetrain.Conv1, binConv(1))
xlabel('weight')
title('Convolution Layer 1')
fprintf('  # of weights in Conv1 = %d\n', numReConv1)

subplot(3,2,2)
hist(WeightRetrain.Conv2, binConv(2))
xlabel('weight')
title('Convolution Layer 2')
fprintf('  # of weights in Conv2 = %d\n', numReConv2)

% Weight Histograms: Convolution layers
binFc = [100, 50, 25];
subplot(3,2,3)
hist(WeightRetrain.Fc1, binFc(1))
xlabel('weight')
title('Fully-connected Layer 1')
fprintf('  # of weights in Fc1 = %d\n', numReFc1)

subplot(3,2,4)
hist(WeightRetrain.Fc2, binFc(2))
xlabel('weight')
title('Fully-connected Layer 2')
fprintf('  # of weights in Fc2 = %d\n', numReFc2)

subplot(3,2,5)
hist(WeightRetrain.Fc3, binFc(3))
xlabel('weight')
title('Fully-connected Layer 3')
fprintf('  # of weights in Fc3 = %d\n', numReFc3)
fprintf('  Retraining Testset Accuracy: %0.3f\n', AccLoss.AccTestRetrain)


%% Pruning results
fprintf('\n* Pruning results')
fprintf('  # of total weights before pruning = %d\n', numPreTotal)
fprintf('  # of total weights after pruning = %d\n', numReTotal)
fprintf('  Compression Rate = %dX\n', round(numPreTotal/numReTotal))