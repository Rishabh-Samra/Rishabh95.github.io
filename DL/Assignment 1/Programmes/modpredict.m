function [losst,pred] = modpredict(test_images,test_labels,W1,W2,W3,W4,b1,b2,b3,b4)
       %one hot encoding  %1fold
       test_l = zeros([length(test_labels),10]);
       for s =1:length(test_labels)
          test_l(s,test_labels(s)+1)=1;
       end
       Ytest = test_l';
       
       losst_ = [];
       
       e=[]; pred = [];
       
       for i = 1:12000 
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
        
       
       end 

        losst = (mean(losst_));
        
        
        
end