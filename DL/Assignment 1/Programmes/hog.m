clear all
clc
test_images=loadMNISTImages('/home/anil/Downloads/MNIST/t10k-images-idx3-ubyte');
test_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/t10k-labels-idx1-ubyte');
load('hogparam.mat');
input_size = 324;
h1_size = 1000;
h2_size = 500;
h3_size = 250;
output_size = 10;
deviation = 0.08;



for i = 1:length(test_images)
    
    image = test_images(:,i);
    img = reshape(image,[28,28]);
    
    
    [hog1,~] = extractHOGFeatures(img,'CellSize',[7 7]);
    
    hog_ = hog1';
    
    [op1,~,~,~] = feedfwd (hog_,W1,W2,W3,W4,b1,b2,b3,b4);
    
    [~,u] = max(op1);

    predict(i) =  u-1;
    
    
    


end
C = confusionmat(test_labels,predict');
acc = 0;
for i=1:size(C,1)
    TP(i)=C(i,i);
    FN(i)=sum(C(i,:))-C(i,i);
    FP(i)=sum(C(:,i))-C(i,i);
    TN(i)=sum(C(:))-TP(i)-FP(i)-FN(i);
    acc = acc+TP(i);
end


P=TP+FN;
N=FP+TN;

Accuracy = acc./length(test_labels);
Sensitivity=TP./P;
Specificity=TN./N;
Precision=TP./(TP+FP);
Recall = TP./(TP+FN);
FPR=1-Specificity;
b=1;
F1_score=( (1+(b^2))*(Sensitivity.*Precision) ) ./ ( (b^2)*(Precision+Sensitivity) );

function [op1,h1,h2,h3] = feedfwd (X,W1,W2,W3,W4,b1,b2,b3,b4)
        a1 = W1*X + b1;
        h1 = (a1>0).*(a1);
    
        a2 = W2*h1 + b2;
        h2 = (a2>0).*(a2);
    
        a3 = W3*h2 + b3;
        h3 = (a3>0).*(a3);   

        a4 = W4*h3 + b4;
        tmp=exp(a4);
        op1=tmp/sum(tmp(:));
        
     
end
visual = [];
for i=1:6
    im = test_images(:,i);
    im_ = reshape(im,[28,28]);
    [~,visual] = extractHOGFeatures(im_,'CellSize',[7 7]);
    imshow(im_);
    hold on
    plot(visual);
end
