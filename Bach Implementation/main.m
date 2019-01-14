%to compare cross-entropy and normal cost function
clear all;
X=[0,0,1;
0,1,1;
1,0,1;
1,1,1];
D=[0;1;1;0];
W1=2*rand(4,3)-1;
W2=2*rand(1,4)-1;
W3=W1;
W4=W2;
J_CF=zeros(1000,1);
J_NM=zeros(1000,1);
for epoch=1:1000
    [W1,W2]=CF_SGD(X,D,W1,W2);
    [W3,W4]=NM_SGD(X,D,W1,W2);
    y2=zeros(4,1);
    h2=zeros(4,1);
    for i=1:4
        v1=W1*X(i,:)';
        y1=sigmoid(v1);
        v2=W2*y1;
        y2(i)=sigmoid(v2);
        g1=W3*X(i,:)';
        h1=sigmoid(g1);
        g2=W4*h1;
        h2(i)=sigmoid(g2);
    end
    J_CF(epoch)=sum((-D.*log(y2)-(1-D).*log(1-y2)),1);
    J_NM(epoch)=sum(((D-h2).^2),1);
end
figure(1);
epoch=1:1000;
plot(epoch,J_CF,'r');
hold on;
plot(epoch,J_NM,'b');
    
    
        
    
