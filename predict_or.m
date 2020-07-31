function [predict_y] = predict_or(os, x)
[label_length,~] = size(x);
alpha = os.alpha;
b = os.b;
SV = find(alpha ~= 0);
temp = alpha(SV, :) .* os.y(SV, :);
k = SMO.Kernel(os.x(SV, :), x);
predict_y = k.' * temp + b * ones(label_length,1);
end