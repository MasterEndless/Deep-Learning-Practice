%single layer neural network_delta rule
clear all;
X=[0,0,1;
    0,1,1;
    1,0,1;
    1,1,1];
D=[0 0 1 1];
W1=2.*rand(1,3)-1;
W2=W1;
W3=W1;
error1=zeros(1,1000);
error2=error1;
error3=error2;
for epoch=1:1000
    W1=SGD_function(W1,X,D);
    W2=batch(W2,X,D);
    W3=minibatch(W3,X,D);
    for k=1:4
        x=X(k,:)';
        v1=W1*x;
        y1(k)=sigmoid(v1);
        v2=W2*x;
        y2(k)=sigmoid(v2);
        v3=W3*x;
        y3(k)=sigmoid(v3);
    end
    error1(epoch)=sum((y1-D).^2);
    error2(epoch)=sum((y2-D).^2);
    error3(epoch)=sum((y3-D).^2);
    
end
epoch=1:1000;
figure(1);
plot(epoch,error1,'r');
hold on;
plot(epoch,error2,'g');
hold on;
plot(epoch,error3,'b');