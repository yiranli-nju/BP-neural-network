%% ===========================清空工作区============================
clc;
clear;
close all;
%% =========================载入数据和模型===========================
load('ps.mat');
load('model.mat');
data = readcell('SI高钾Day-5.xlsx');
XX = data(2:end,2:end);
X = cell2mat(XX);
%% ======================利用训练完毕的模型预测=======================
input0 = X';                   % 输入模型的训练集特征
input = mapminmax('apply',input0,ps);
Score = sim(model,input);
pre_Y = vec2ind(Score);
pre_Y = pre_Y';
Score = Score'; 
%% =============================绘制预测折线图=======================
%-----预测折线图------
figure()
plot(pre_Y,'r-*');
xlabel('测试样本序号');
ylabel('预测类别');
title('预测结果');
ylim([0.5,5.5])
set(gca,'YTick',1:5)
set(gca,'YTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
grid on
box on
%-----预测结果柱状图---
count_num = zeros(5,1);
x = 1:1:5;
for i = x
    ind = find(pre_Y==i);
    count_num(i) = length(ind);
end
figure()
bar(1:1:5,count_num)
xlabel('预测类别');
ylabel('样本数量');
grid on
box on
set(gca,'xtick',1:1:5,'XTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'}, ...
    'FontSize',10);
for i = 1:length(count_num)
    text(x(i),count_num(i), num2str(count_num(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end
%% ============================结果写出============================
out1 = [{'预测类别'};num2cell(pre_Y)];
out = [data,out1];
out{1,1} = nan;
writecell(out,'预测结果导出.xlsx');


