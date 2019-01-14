function [ W1,W2,W3,W4,W5,W6 ] = BP_SGD( W1,W2,W3,W4,W5,W6,X,D )
    alpha=0.9;beta=0.9;
    mmt1=zeros(size(W1));
    mmt2=zeros(size(W2));
    mmt3=zeros(size(W3));
    mmt4=zeros(size(W4));
    mmt5=zeros(size(W5));
    mmt6=zeros(size(W6));
    
    for k=1:4
        x=X(k,:)';
        d=D(k);
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
        y=sigmoid_function(v);
        e=d-y;
        delta=y.*(1-y)*e;
        e1=W6'*delta;
        delta1=y5.*(1-y5).*e1;
        e2=W5'*delta1;
        delta2=y4.*(1-y4).*e2;
        e3=W4'*delta2;
        delta3=y3.*(1-y3).*e3;
        e4=W3'*delta3;
        delta4=y2.*(1-y2).*e4;
        e5=W2'*delta4;
        delta5=y1.*(1-y1).*e5;
        
        dW1=alpha*delta5*x';
        mmt1=dW1+mmt1;
        W1=W1+mmt1;
        
        dW2=alpha*delta4*y1';
        mmt2=dW2+mmt2;
        W2=W2+mmt2;
        
        dW3=alpha*delta3*y2';
        mmt3=dW3+mmt3;
        W3=W3+dW3;
        
        dW4=alpha*delta2*y3';
        mmt4=mmt4+dW4;
        W4=W4+mmt4;
        
        dW5=alpha*delta1*y4';
        mmt5=dW5+mmt5;
        W5=W5+mmt5;
        
        dW6=alpha*delta*y5';
        mmt6=mmt6+dW6;
        W6=W6+mmt6;
    end
        
end

