function [X_test, Y_test, X_train, Y_train] = split(x, y, rate)
    % x��ԭ���ݼ����ֳ�ѵ�������Ͳ������� 
    [ndata, ~] = size(x);        %ndata��������Dά��
    num_test = floor(ndata * rate);
    R = randperm(ndata);         %1��n��Щ��������ҵõ���һ���������������Ϊ����
    X_test = x(R(1:num_test), :);  %��������ǰ���ݵ���Ϊ��������Xtest
    Y_test = y(R(1:num_test), :);
    R(1:num_test) = [];
    X_train = x(R, :);          %ʣ�µ�������Ϊѵ������Xtrain
    Y_train = y(R, :);
end