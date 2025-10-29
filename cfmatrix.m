function [confmatrix] = cfmatrix(actual, predict)
% actual 真实值
% predict 预测值
% classlist 类别列表

classlist = unique(actual);
format short g;
n_class = length(classlist);
for i = 1:n_class
    obind_class_i = find(actual == classlist(i));
    prind_class_i = find(predict == classlist(i));
    confmatrix(i,i) = length(intersect(obind_class_i,prind_class_i));
    for j = 1:n_class
        %if (j ~= i)
        if (j < i)
        % observed j predicted i
        confmatrix(i,j) = length(find(actual(prind_class_i) == classlist(j))); 
        % observed i predicted j
        confmatrix(j,i) = length(find(predict(obind_class_i) == classlist(j)));
        end
    end
    
end
