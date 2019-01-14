%BP alogorithm _ 5 layers
X=[0,0,1;
    0,1,1;
    1,0,1;
    1,1,1];
D=[0,1,1,0];
W1=3*rand(4,3)-1;
W2=3*rand(4,4)-1;
W3=3*rand(4,4)-1;
W4=3*rand(4,4)-1;
W5=3*rand(4,4)-1;
W6=3*rand(1,4)-1;
for epoch=1:25000
    [W1,W2,W3,W4,W5,W6]=BP_SGD(W1,W2,W3,W4,W5,W6,X,D);
end
for k=1:4
        x=X(k,:)';
        v1=W1*x;
        y1=sigmoid_function(v1);
        v2=W2*y1;
        y2=sigmoid_function(v2);
        v3=W3*y2;
        y3=sigmoid_function(v3);
        v4=W4*y3;
        y4=sigmoid_function(v4);
        v5=W5*y4;
        y5=sigmoid_function(v5);
        v=W6*y5;
        y=sigmoid_function(v)
end

    
