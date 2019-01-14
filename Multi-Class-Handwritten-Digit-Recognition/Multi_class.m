function [ W1,W2 ] = Multi_class( X,D,W1,W2 )
    alpha=0.9;beta=0.9;
    m1=zeros(size(W1));
    m2=zeros(size(W2));
    for k=1:4
        x=reshape(X(:,:,k),25,1);
        d=D(:,k);
        v1=W1*x;
        y1=sigmoid(v1);
        v2=W2*y1;
        y2=softmax(v2);
        e=d-y2;
        delta=e;
        e1=W2'*delta;
        delta1=e1.*(1-y1).*y1;
        dW1=alpha*delta1*x';
        m1=beta.*m1+dW1;
        W1=W1+m1;
        dW2=alpha*delta*y1';
        m2=beta.*m2+dW2;
        W2=W2+m2;
    end


end

