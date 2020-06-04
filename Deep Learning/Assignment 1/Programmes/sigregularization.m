clc
close all
clear all

rng('default')    % For reproducibility
 
load('Augment_data.mat')

%% define inputs
%train_images=loadMNISTImages('/home/anil/Downloads/MNIST/train-images-idx3-ubyte');
%train_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/train-labels-idx1-ubyte');
test_images=loadMNISTImages('/home/anil/Downloads/MNIST/t10k-images-idx3-ubyte');
test_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/t10k-labels-idx1-ubyte');


%% defining structure
input_size = 784;
h1_size = 1000;
h2_size = 500;
h3_size = 250;
output_size = 10;
deviation = 0.08;
bs = 64;
lr=0.01;
m=0.9;

W1 = normrnd(0,deviation,[h1_size,input_size]);
W2 = normrnd(0,deviation,[h2_size,h1_size]);
W3 = normrnd(0,deviation,[h3_size,h2_size]);
W4 = normrnd(0,deviation,[output_size,h3_size]);
b1 = normrnd(0,deviation,[h1_size,1]);
b2 = normrnd(0,deviation,[h2_size,1]);
b3 = normrnd(0,deviation,[h3_size,1]);
b4 = normrnd(0,deviation,[output_size,1]);



vec_loc=[1:60000];
for i =1:8
     vec_loc =[vec_loc, randperm(60000)];
end
iterations=8000;
r=1;
 

 losst=[];   
 %%training
for i = 1:iterations                               %iterations start
    

    
    db4_agg = zeros(output_size,1);
    db3_agg = zeros(h3_size,1);
    db2_agg = zeros(h2_size,1);
    db1_agg = zeros(h1_size,1);

    dw4_agg = zeros(output_size,h3_size);
    dw3_agg = zeros(h3_size,h2_size);
    dw2_agg = zeros(h2_size,h1_size);
    dw1_agg = zeros(h1_size,input_size);
    
    
    v4 = zeros(size(W4));
    v3 = zeros(size(W3));
    v2 = zeros(size(W2));
    v1 = zeros(size(W1));

    v4_= zeros(size(b4));
    v3_= zeros(size(b3));
    v2_= zeros(size(b2));
    v1_= zeros(size(b1));

    
    
    j=bs*(i-1)+1;
    k=bs*i;
    image_batch=train_img_aug(:,vec_loc(j:k));
    label_batch=train_labels_aug(vec_loc(j:k))';
    %one hot encoding
    y_train = zeros(length(label_batch),10);
    for s = 1:size(y_train,1)
        y_train(s,label_batch(s)+1)=1;
    end

    
    
    loss=0;
    for c = 1: bs                                      %batch data
        X= image_batch(:,c);
        Y= (y_train(c,:))';
        [op1,h1,h2,h3] = feedfwd (X,W1,W2,W3,W4,b1,b2,b3,b4);
        
        %cross entropy loss
        loss = loss-(sum(Y.*log(op1)));
        
        %sum_loss(i) = sum_loss(i)+loss;  %60 img loss
        
       
%backprop
        dl4 =  (op1-Y);
        db4_agg = (db4_agg) + dl4;
        dw4_agg = (dw4_agg) + dl4*h3';
        
        dl3 = W4' * dl4 .* (h3>0);
        db3_agg = (db3_agg) + dl3;
        dw3_agg = (dw3_agg) + dl3*h2';
        
        dl2 = W3' * dl3 .* (h2>0);
        db2_agg = (db2_agg) + dl2;
        dw2_agg = (dw2_agg) + dl2*h1';
        
        dl1 = W2' * dl2 .* (h1>0);
        db1_agg = (db1_agg) + dl1;
        dlw1 = (dw1_agg) + dl1 * X';
    end
    avg_loss(i)=loss/bs;
    
   
    if mod(i,100)==0
    disp(['Loss @', num2str(i), 'iter:', num2str(avg_loss(i))]) 
    end
    
    %update on h3
    grad_w4 = (dw4_agg/bs)+(0.005*W4);
    v4 = m*v4 + lr* grad_w4;
    W4 = W4 - v4;
    
    grad_b4 = db4_agg/bs;
    v4_ = m*v4_ + lr*grad_b4;
    b4 = b4 - v4_;
    
    %update on h2
    grad_w3 = (dw3_agg/bs)+(0.005*W3);
    v3 = m*v3 + lr* grad_w3;
    W3 = W3 -v3;
    
    grad_b3 = db3_agg/bs;
    v3_ = m*v3_ + lr*grad_b3;
    b3 = b3 - v3_;
    
    %update on h1
    grad_w2 = (dw2_agg/bs)+(0.005*W2);
    v2 = m*v2 + lr* grad_w2;
    W2 = W2 - v2;
    
    grad_b2 = db2_agg/bs;
    v2_ = m*v2_ + lr*grad_b2;
    b2 = b2 - v2_;
    
     %update on inp layer
    grad_w1 = (dw1_agg/bs)+(0.005*W1);
    v1 = m*v1 + lr*grad_w1;
    W1 = W1 - v1;
    
    grad_b1 = db1_agg/bs;
    v1_ = m*v1_ + lr*grad_b1;
    b1 = b1 - v1_;
    
    if mod(i,200)==0
        [losst(r),pred] =relumodpred(test_images,test_labels,W1,W2,W3,W4,b1,b2,b3,b4);
        disp(['Test Loss after',num2str(i),'iterations is',num2str(losst(r))]);
        r=r+1;
    end
    
end    
  
iter = 1:iterations;
plot(iter, avg_loss);

plot(losst);

 %confusion matrix
        actual = test_labels';
        predd = pred';
        C = confusionmat(test_labels,predd);
        acc = 0;
        for i=1:size(C,1)
             TP(i)=C(i,i);
             FN(i)=sum(C(i,:))-C(i,i);
             FP(i)=sum(C(:,i))-C(i,i);
             TN(i)=sum(C(:))-TP(i)-FP(i)-FN(i);
             acc = acc+TP(i);
        end
% P and N are the total number of actual Positive and negative samples
        P=TP+FN;
        N=FP+TN;
        Accuracy = acc/length(test_images);
        Sensitivity=TP./P;
        Specificity=TN./N;
        Precision=TP./(TP+FP);
        Recall = TP./(TP+FN);
        FPR=1-Specificity;
        beta=1;
        F1_score=( (1+(beta^2))*(Sensitivity.*Precision) ) ./ ( (beta^2)*(Precision+Sensitivity) );

        save('Regularization','W1','W2','W3','W4','b1','b2','b3','b4','C','Accuracy','Precision','Recall','avg_loss','losst')






function [op1,h1,h2,h3] = feedfwd (X,W1,W2,W3,W4,b1,b2,b3,b4)
        a1 = W1*X + b1;
        h1 = max(a1,0);
    
        a2 = W2*h1 + b2;
        h2 = max(a2,0);
    
        a3 = W3*h2 + b3;
        h3 = max(a3,0);   

        a4 = W4*h3 + b4;
        tmp = exp(a4);
        op1 = exp(a4)/sum(tmp);
       
end





