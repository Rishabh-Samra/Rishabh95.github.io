clc
close all
clear all
function [C,Sensitivity,Precision,Recall,FPR,F1_score,Accuracy]=confmat(test_labels,pred)

C = confusionmat(test_images,pred');
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

Accuracy = acc./length(test_labels)
Sensitivity=TP./P;
Specificity=TN./N;
Precision=TP./(TP+FP);
Recall = TP./(TP+FN);
FPR=1-Specificity;
b=1;
F1_score=( (1+(b^2))*(Sensitivity.*Precision) ) ./ ( (b^2)*(Precision+Sensitivity) );