clear all
clc
test_images=loadMNISTImages('/home/anil/Downloads/MNIST/t10k-images-idx3-ubyte');
test_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/t10k-labels-idx1-ubyte');
load('kf2trainsig.mat');
diary('Testlosskfold')
test_l = zeros([length(test_labels),10]);
       for s =1:length(test_labels)
          test_l(s,test_labels(s)+1)=1;
       end
       Ytest = test_l';
       
       losst_ = [];
       
       e=[]; pred = [];
       load('kf2trainsig.mat');
       
     for i = 1:10000 
        A1 = W1*test_images(:,i) + b1;
        H1 = 1./(1+exp(-A1));
    
        A2 = W2*H1 + b2;
        H2 = 1./(1+exp(-A2));
    
        A3 = W3*H2 + b3;
        H3 = 1./(1+exp(-A3));   

        A4 = W4*H3 + b4;
        tmp=exp(A4);
        out1=tmp/sum(tmp(:));
        [~,e(i)] = max(out1);
        pred = [pred , e(i)-1];
        
        losst_(i) = -sum(Ytest(:,i).*log(out1));
        if mod(i,200)==0
            disp(['Loss',num2str(i),'iterations',num2str(losst_(i))]);
        end  
            
     end
        
        
        
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
        Accuracy = acc/length(test_labels);
        Sensitivity=TP./P;
        Specificity=TN./N;
        Precision=TP./(TP+FP);
        Recall = TP./(TP+FN);
        FPR=1-Specificity;
        beta=1;
        F1_score=( (1+(beta^2))*(Sensitivity.*Precision) ) ./ ( (beta^2)*(Precision+Sensitivity) );


loss = mean(losst_)
        