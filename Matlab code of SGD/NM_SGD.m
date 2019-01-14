function [ W3,W4 ] = NM_SGD( X,D,W3,W4 )
    alpha=0.9;beta=0.9;
    m1=zeros(size(W3));
    m2=zeros(size(W4));
    for i=1:4
        x=X(i,:);
        d=D(i);
        v1=W3*x';
        y1=sigmoid(v1);
        v2=W4*y1;
        y2=sigmoid(v2);
        e=d-y2;
        delta=e.*(1-y2).*y2;
        e1=W4'*delta;
        delta1=e1.*(1-y1).*y1;
        dW3=alpha*delta1*x;
        m1=beta.*m1+dW3;
        W3=W3+m1;
        dW4=alpha*delta*y1';
        m2=beta.*m2+dW4;
        W4=W4+m2;
    end


end

