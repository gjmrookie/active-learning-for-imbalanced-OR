function [L_X, L_Y, U_X, U_Y] = ModifyData4Semi(x, y)
    global unlabel_size or_class_number
    [sample_numbers, ~] = size(y);
    L_index = [];
    label_number = floor((sample_numbers - unlabel_size) / or_class_number);
    for label = 1:or_class_number
        label_index = find(y == label);
        if(length(label_index) < label_number)
            L_index = [L_index; label_index];
        else
            L_index = [L_index; label_index(1:label_number)];
        end
    end
    U_index = 1:sample_numbers;
    U_index(L_index) = [];
    L_X = x(L_index, :);
    L_Y = y(L_index, :);
    U_X = x(U_index, :);
    U_Y = y(U_index, :);
end