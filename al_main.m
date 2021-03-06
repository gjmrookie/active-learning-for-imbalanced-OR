function [err_zo_list, err_ab_list, err_im_list, err_re_list, SV_list, number_train_size] = al_main(data_flag)
    global or_class_number C unlabel_size
    [x,y] = read_data(data_flag);
    [or_class_number, ~] = size(unique(y));
    ge = floor(unlabel_size / or_class_number) - 1;
    [X_test, Y_test, X_train, Y_train] = split(x, y, 0.2);
    [L_X, L_Y, U_X, U_Y] = ModifyData4Semi(X_train, Y_train);
    %extend the data
    [L_X_extend, L_Y_extend] = extend_binary(L_X, L_Y);
    [U_X_extend, U_Y_extend] = extend_binary(U_X, U_Y);
    [X_test_extend, Y_test_extend] = extend_binary(X_test, Y_test);
    %get a model
    SV_list = [];
    tic;
    os = SMO.InitialSolution(L_X_extend, L_Y_extend, C);
    SV = length(find(os.alpha ~= 0));
    SV_list = [SV_list, SV];
    [predict_y] = predict_or(os, X_test_extend);
    err_zo_list = [];
    err_ab_list = [];
    err_im_list = [];
    err_re_list = [];
    [err_zo, err_ab, err_im, err_re] = get_accuracy(predict_y, Y_test);
    err_zo_list = [err_zo_list, err_zo];
    err_ab_list = [err_ab_list, err_ab];
    err_im_list = [err_im_list, err_im];
    err_re_list = [err_re_list, err_re];
    number_train_size = [];
    number_train_size = [number_train_size, size(L_X, 1)];
    for g = 1:ge
        [predict_y] = predict_or(os, U_X_extend);
        [label_length, ~] = size(predict_y);
        temp_y = predict_y(1:label_length);
        predict_y(find(predict_y < 0)) = -1;
        predict_y(find(predict_y >= 0)) = 1;
        final_y = [];
        for i = 1:label_length/(or_class_number-1)
            sum_y = 1;
            for j = 1:or_class_number-1
                if predict_y((i-1) * (or_class_number-1) + j) == 1
                    sum_y = sum_y + 1;
                end
            end
            final_y = [final_y; sum_y];
        end
        confidence = [];
        for i = 1:label_length/(or_class_number-1)
            for j = 1:or_class_number - 2
                if predict_y((i-1) * (or_class_number-1) + 1) == -1
                    confidence = [confidence; abs(temp_y((i-1) * (or_class_number-1) + 1))];
                    break;
                elseif predict_y((i-1) * (or_class_number-1) + j) == 1 && predict_y((i-1) * (or_class_number - 1) + j + 1) == -1
                    confidence = [confidence; min(abs(temp_y((i-1) * (or_class_number-1) + j)), abs(temp_y((i-1) * (or_class_number - 1) + j + 1)))];
                    break;
                elseif predict_y(i * (or_class_number - 1)) == 1
                    confidence = [confidence; abs(temp_y((i - 1) * (or_class_number - 1) + or_class_number - 1))];
                    break;
                end
            end
        end
        for i = 1:or_class_number
            label_index = find(final_y == i);
            label_confidence = confidence(label_index);
            [M, I] = min(label_confidence);
            L_X = [L_X; U_X(I, :)];
            U_X(I, :) = [];
            L_Y = [L_Y; U_Y(I)];
            U_Y(I) = [];
            move_index = [];
            for k = 1:or_class_number - 1
                L_X_extend = [L_X_extend; U_X_extend((I - 1) * (or_class_number - 1) + k, :)];
                L_Y_extend = [L_Y_extend; U_Y_extend((I - 1) * (or_class_number - 1) + k, :)];
                move_index = [move_index; (I - 1) * (or_class_number - 1) + k];
            end
            U_X_extend(move_index, :) = [];
            U_Y_extend(move_index, :) = [];
        end
        os = SMO.InitialSolution(L_X_extend, L_Y_extend, C);
        SV = length(find(os.alpha ~= 0));
        SV_list = [SV_list, SV];
        [predict_y] = predict_or(os, X_test_extend);
        [err_zo, err_ab, err_im, err_re] = get_accuracy(predict_y, Y_test);
        err_zo_list = [err_zo_list, err_zo];
        err_ab_list = [err_ab_list, err_ab];
        err_im_list = [err_im_list, err_im];
        err_re_list = [err_re_list, err_re];
        number_train_size = [number_train_size, size(L_X, 1)];
    end
end