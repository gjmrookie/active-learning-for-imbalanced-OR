function [X_extend, Y_extend] = extend_binary(X, Y)
    %labeled data
    global or_class_number
    [X_len, ~] = size(Y);
    X_extend = [];
    Y_extend = [];
    for i = 1:X_len
        for j = 1:or_class_number - 1
            ek = zeros(1, or_class_number - 1);
            ek(j) = ek(j) +1;
            x_new = [X(i, :), ek];
            X_extend = [X_extend; x_new];
            if j < Y(i)
                y_i_k = 1;
            else
                y_i_k = -1;
            end
            Y_extend = [Y_extend; y_i_k];
        end
    end
end