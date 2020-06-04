train_images=loadMNISTImages('/home/anil/Downloads/MNIST/train-images-idx3-ubyte');
train_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/train-labels-idx1-ubyte');
test_images=loadMNISTImages('/home/anil/Downloads/MNIST/t10k-images-idx3-ubyte');
test_labels=loadMNISTLabels('/home/anil/Downloads/MNIST/t10k-labels-idx1-ubyte');

train_img_aug = [];
train_labels_aug = [];

for i = 1:60000
    p = train_images(:,i);
    q = reshape(p,[28,28]);
    angle_rotate = randi([-20,20]);
    r = imrotate(q,angle_rotate,'crop');        
    s = r(:);
    
     train_img_aug = [train_img_aug, p(:),s];
     train_labels_aug = [train_labels_aug; train_labels(i); train_labels(i)];
end
    
