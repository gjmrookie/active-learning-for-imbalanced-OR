function [err_zo_list, err_ab_list, err_im_list, number_train_size] = al_com(data_flag, ge)
    global or_class_number C C_1
    [x,y] = read_data(data_flag);
    [or_class_number, ~] = size(unique(y));
    [X_test, Y_test, X_train, Y_train] = split(x, y, 0.2);
    [L_X, L_Y, U_X, U_Y] = ModifyData4Semi(X_train, Y_train);
    %extend the data
    [L_X_extend, L_Y_extend] = extend_binary(L_X, L_Y);
    [U_X_extend, U_Y_extend] = extend_binary(U_X, U_Y);
    [X_test_extend, Y_test_extend] = extend_binary(X_test, Y_test);
    %get a model
    tic;
    os_1 = SMO.InitialSolution(L_X_extend, L_Y_extend, C);
	os_2 = SMO.InitialSolution(L_X_extend, L_Y_extend, C_1);
    [predict_y_1] = predict_or(os_1, X_test_extend);
    [predict_y_2] = predict_or(os_2, X_test_extend);
    err_zo_list = [];
    err_ab_list = [];
    err_im_list = [];
    [err_zo_1, err_ab_1, err_im_1] = get_accuracy(predict_y_1, Y_test);
    [err_zo_2, err_ab_2, err_im_2] = get_accuracy(predict_y_2, Y_test);
    err_zo_list = [err_zo_list, (err_zo_1 + err_zo_2) / 2];
    err_ab_list = [err_ab_list, (err_ab_1 + err_ab_2) / 2];
    err_im_list = [err_im_list, (err_im_1 + err_im_2) / 2];
    number_train_size = [];
    number_train_size = [number_train_size, size(L_X, 1)];
    for g = 1:ge
        [predict_y_1] = predict_or(os_1, U_X_extend);
        [predict_y_2] = predict_or(os_2, U_X_extend);
        [label_length, ~] = size(predict_y_1);
        temp_y_1 = predict_y_1(1:label_length);
        predict_y_1(find(predict_y_1 < 0)) = -1;
        predict_y_1(find(predict_y_1 >= 0)) = 1;
        final_y_1 = [];
        for i = 1:label_length/(or_class_number-1)
            sum_y = 1;
            for j = 1:or_class_number-1
                if predict_y_1((i-1) * (or_class_number-1) + j) == 1
                    sum_y = sum_y + 1;
                end
            end
            final_y_1 = [final_y_1; sum_y];
        end
        temp_y_2 = predict_y_2(1:label_length);
        predict_y_2(find(predict_y_2 < 0)) = -1;
        predict_y_2(find(predict_y_2 >= 0)) = 1;
        final_y_2 = [];
        for i = 1:label_length/(or_class_number-1)
            sum_y = 1;
            for j = 1:or_class_number-1
                if predict_y_2((i-1) * (or_class_number-1) + j) == 1
                    sum_y = sum_y + 1;
                end
            end
            final_y_2 = [final_y_2; sum_y];
        end
        confidence = [];
        for i = 1:label_length/(or_class_number-1)
            for j = 1:or_class_number - 2
                if predict_y_1((i-1) * (or_class_number-1) + 1) == -1
                    confidence = [confidence; abs(temp_y_1((i-1) * (or_class_number-1) + 1))];
                    break;
                elseif predict_y_1((i-1) * (or_class_number-1) + j) == 1 && predict_y_1((i-1) * (or_class_number - 1) + j + 1) == -1
                    confidence = [confidence; min(abs(temp_y_1((i-1) * (or_class_number-1) + j)), abs(temp_y_1((i-1) * (or_class_number - 1) + j + 1)))];
                    break;
                elseif predict_y_1(i * (or_class_number - 1)) == 1
                    confidence = [confidence; abs(temp_y_1((i - 1) * (or_class_number - 1) + or_class_number - 1))];
                    break;
                end
            end
        end
        for add_number = 1:10
            label_index = find(final_y_1 ~= final_y_2);
            if(size(label_index, 1) == 0)
                break;
            end
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
        os_1 = SMO.InitialSolution(L_X_extend, L_Y_extend, C);
        os_2 = SMO.InitialSolution(L_X_extend, L_Y_extend, C_1);
        [predict_y_1] = predict_or(os_1, X_test_extend);
        [predict_y_2] = predict_or(os_2, X_test_extend);
        [err_zo_1, err_ab_1, err_im_1] = get_accuracy(predict_y_1, Y_test);
        [err_zo_2, err_ab_2, err_im_2] = get_accuracy(predict_y_2, Y_test);
        err_zo_list = [err_zo_list, (err_zo_1 + err_zo_2) / 2];
        err_ab_list = [err_ab_list, (err_ab_1 + err_ab_2) / 2];
        err_im_list = [err_im_list, (err_im_1 + err_im_2) / 2];
        number_train_size = [number_train_size, size(L_X, 1)];
    end
end