clear all
close all
clc
train_images=loadMNISTImages('/home/anil/Downloads/MNIST/train-images-idx3-ubyte');
train_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/train-labels-idx1-ubyte');
test_images=loadMNISTImages('/home/anil/Downloads/MNIST/t10k-images-idx3-ubyte');
test_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/t10k-labels-idx1-ubyte');
% X = train_images';
Mdl = fitcknn(train_images',train_labels,'NumNeighbors',5,'Standardize',1);
% X_ = test_images';

pred = predict(Mdl,test_images');


C = confusionmat(test_labels,pred);
acc = 0;

for i = 1:10
    acc = acc+C(i,i);
end

accuracy = acc/10000;
 

 
 