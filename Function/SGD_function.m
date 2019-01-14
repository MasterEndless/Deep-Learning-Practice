function [ W ] = SGD_function( W,X,D )
    alpha=0.9;N=4;
    for k=1:N
        x=X(k,:)';
        d=D(k);
        v=W*x;
        y=sigmoid_function(v);
        e=d-y;
        delta=y*(1-y)*e;
        dW=alpha*delta*x';
        W=W+dW;
    end
end

