function [err_zo, err_ab, err_im, err_re, Amae, Mmae] = get_accuracy(predict_y, Y)
    global or_class_number
    [label_length, ~] = size(predict_y);
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
    [Y_length, ~] = size(Y);
    [err_length, ~] = size(find(final_y - Y ~= 0));
    err_zo = err_length / Y_length;
    err_ab = sum(abs(final_y - Y)) / Y_length;
    ab_sum = 0;
    for i = 1:or_class_number
        label_index = find(Y == i);
        label_length = size(label_index, 1);
        if label_length == 0
            continue;
        end
        ab_sum = ab_sum + (Y_length / label_length) * sum(abs(final_y(label_index, :) - Y(label_index, :)));
    end
    err_im = ab_sum / Y_length;
    err_im = err_im / or_class_number;
    err_all = 0;
    count = 0;
    for i = 1:or_class_number
        label_index = find(Y == i);
        if(label_index > 0)
           count = count + 1;
           [err_length, ~] = size(find(final_y(label_index, :) - Y(label_index, :) == 0));
           err_all = err_all + err_length / length(label_index);
        end
    end
    err_re = err_all / count;
    mae_list = [];
    for i = 1:or_class_number
        label_index = find(Y == i);
        if length(label_index) == 0
            continue;
        end
        err_l = sum(abs(final_y(label_index) - Y(label_index))) / length(label_index);
        mae_list = [mae_list, err_l];
    end
    Amae = mean(mae_list);
    Mmae = max(mae_list);
end