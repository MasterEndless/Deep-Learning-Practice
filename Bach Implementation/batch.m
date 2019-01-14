function [ W ] = batch( W,X,D )
    alpha=0.9;
    dW=zeros(4,3);
    for k=1:4
        x=X(k,:)';
        d=D(k);
        v=W*x;
        y=sigmoid_function(v);
        e=d-y;
        delta=e*y*(1-y);
        dW(k,:)=alpha*delta*x';
    end
    W=W+sum(dW)/4;
        
end

