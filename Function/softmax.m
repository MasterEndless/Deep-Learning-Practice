function [ Y ] = softmax( X )
    ex=exp(X);
    Y=ex./(sum(ex));
end

