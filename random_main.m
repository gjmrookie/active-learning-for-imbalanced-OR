function [err_zo_list, err_ab_list, err_im_list, err_re_list, SV_list, number_train_size, err_am_list, err_mm_list] = random_main(data_flag, ge)
    global or_class_number C
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
    SV_list = [];
    os = SMO.InitialSolution(L_X_extend, L_Y_extend, C);
    SV = length(find(os.alpha ~= 0));
    SV_list = [SV_list, SV];
    [predict_y] = predict_or(os, X_test_extend);
    err_zo_list = [];
    err_ab_list = [];
    err_im_list = [];
    err_re_list = [];
    err_am_list = [];
    err_mm_list = [];
    [err_zo, err_ab, err_im, err_re, err_am, err_mm] = get_accuracy(predict_y, Y_test);
    err_zo_list = [err_zo_list, err_zo];
    err_ab_list = [err_ab_list, err_ab];
    err_im_list = [err_im_list, err_im];
    err_re_list = [err_re_list, err_re];
    err_am_list = [err_am_list, err_am];
    err_mm_list = [err_mm_list, err_mm];
    number_train_size = [];
    number_train_size = [number_train_size, size(L_X, 1)];
    for g = 1:ge
        for add_sample = 1:10
            [U_length, ~] = size(U_X);
            I = randi(U_length);
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
        [err_zo, err_ab, err_im, err_re, err_am, err_mm] = get_accuracy(predict_y, Y_test);
        err_zo_list = [err_zo_list, err_zo];
        err_ab_list = [err_ab_list, err_ab];
        err_im_list = [err_im_list, err_im];
        err_re_list = [err_re_list, err_re];
        err_am_list = [err_am_list, err_am];
        err_mm_list = [err_mm_list, err_mm];
        number_train_size = [number_train_size, size(L_X, 1)];
    end
end