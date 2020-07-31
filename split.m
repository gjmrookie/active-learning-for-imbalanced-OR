function [X_test, Y_test, X_train, Y_train] = split(x, y, rate)
    % x是原数据集，分出训练样本和测试样本 
    [ndata, ~] = size(x);        %ndata样本数，D维数
    num_test = floor(ndata * rate);
    R = randperm(ndata);         %1到n这些数随机打乱得到的一个随机数字序列作为索引
    X_test = x(R(1:num_test), :);  %以索引的前数据点作为测试样本Xtest
    Y_test = y(R(1:num_test), :);
    R(1:num_test) = [];
    X_train = x(R, :);          %剩下的数据作为训练样本Xtrain
    Y_train = y(R, :);
end