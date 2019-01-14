function [ W1,W2 ] = CF_SGD( X,D,W1,W2 )
    alpha=0.9;beta=0.9;
    m1=zeros(size(W1));
    m2=zeros(size(W2));
    for i=1:4
        x=X(i,:);
        d=D(i);
        v1=W1*x';
        y1=sigmoid(v1);
        v2=W2*y1;
        y2=sigmoid(v2);
        e=d-y2;
        delta=e;
        e1=W2'*delta;
        delta1=e1.*(1-y1).*y1;
        dW1=alpha*delta1*x;
        m1=beta.*m1+dW1;
        W1=W1+m1;
        dW2=alpha*delta*y1';
        m2=beta.*m2+dW2;
        W2=W2+m2;
    end
        
        
end

