%% ===========================清空工作区============================
clc;
clear;
close all;
%% ============================载入数据=============================
data0 = xlsread('test-8.xlsx');
data = data0(:,2:end);
X = [];
label = [];
for i = 1:1:size(data,2)/5
    tmp = data(:,(i-1)*5+1:5*i);
    X = [X;tmp];                          % 原始特征 
    label = [label;i*ones(size(tmp,1),1)];  % 原始标签
end
%---------------打乱数据-------------
N = length(label);           % 样本数
IND1 = randperm(N);          % 置乱序列
X = X(IND1,:);               % 打乱后的特征数据
label = label(IND1,:);       % 打乱后的标签数据
%% ========================划分训练集和测试集========================
%---------------------------
K = 0.8;                     % 训练样本占用的比重
class = unique(label);
train_X = [];                % 训练集特征
train_label = [];            % 训练集标签
test_X = [];                 % 测试集特征
test_label = [];             % 测试集标签
for i = 1:1:length(class)
    index = find(label==class(i));
    tempX = X(index,:);
    tempL = label(index,:);
    tempN = length(tempL);
    train_X = [train_X;tempX(1:ceil(K*tempN),:)];
    train_label = [train_label;tempL(1:ceil(K*tempN),1)];
    test_X = [test_X;tempX(ceil(K*tempN)+1:end,:)];
    test_label = [test_label;tempL(ceil(K*tempN)+1:end,1)];
end
input_train = train_X';                   % 输入模型的训练集特征
output_train = ind2vec(train_label');     % 输入模型的训练集输出
input_test = test_X';                     % 输入模型的测试集特征
output_test = ind2vec(test_label');       % 输入模型的测试集输出
%% ======================训练特征和测试特征归一化=====================
[input_train,ps] = mapminmax(input_train,0,1);
input_test = mapminmax('apply',input_test,ps);
%% =========================创建BP神经网络==========================
hiddennum = [10]; % 隐含层设置
net = newff(input_train,output_train,hiddennum, ...
           {'tansig','purelin'},'trainlm'); % 建立网络模型，传递函数使用purelin，采用梯度下降法训练
                                            % { 'logsig' 'purelin' } , 'traingdx'
%% ========================设置训练参数==============================
net.trainParam.showWindow = true;      % 出现训练窗口
net.trainParam.showCommandLine = true; % 出现训练窗口
net.trainparam.show = 50 ;        % 每50epoch更新一次信息
net.trainparam.epochs = 500 ;     % 训练周期
net.trainparam.goal = 1E-6 ;      % 训练目标
net.trainParam.lr = 0.01 ;        % 学习率
%% ===========================BP训练================================
[model,trace] = train(net,input_train,output_train);
plotperform(trace);   % 绘制训练过程
%% ==========================模型测试===============================
Score = sim(model,input_test);
Score1 = sim(model,input_train);
pre_Y = vec2ind(Score);
pre_Y1 = vec2ind(Score1);
pre_Y = pre_Y';   pre_Y1 = pre_Y1';
Score = Score';   Score1 = Score1';
diff = test_label-pre_Y;
corr_num = length(find(diff==0));
acc = corr_num/length(diff);
diff1 = train_label-pre_Y1;
corr_num1 = length(find(diff1==0));
acc1 = corr_num1/length(diff1);
fprintf('训练集预测正确率是：%.4f\n',acc1);
fprintf('测试集预测正确率是：%.4f\n',acc);
%% =============================绘制折线图==========================
%-----训练集------
figure()
plot(train_label,'r-*');
hold on
plot(pre_Y1,'b-o');
xlabel('测试样本序号');
ylabel('类别');
legend('真实类别','预测类别');
title(['训练集预测结果',num2str(100*acc1),'%']);
set(gca,'YTick',0:5)
set(gca,'YTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
%-----测试集------
figure()
plot(test_label,'r-*');
hold on
plot(pre_Y,'b-o');
xlabel('测试样本序号');
ylabel('类别');
legend('真实类别','预测类别');
title(['测试集预测结果',num2str(100*acc),'%']);
set(gca,'YTick',1:5)
set(gca,'YTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
%% ============================绘制混淆矩阵=========================
figure();
Y1 = categorical(train_label');
Y2 = categorical(pre_Y1');
plotconfusion(Y1,Y2)
title(['训练集准确率=',num2str(100*acc1),'%'],'Fontsize',12);
set(gca,'YTick',1:5)
set(gca,'YTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
set(gca,'XTick',1:5)
set(gca,'XTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
figure();
Y1 = categorical(test_label');
Y2 = categorical(pre_Y');
plotconfusion(Y1,Y2)
title(['测试集准确率=',num2str(100*acc),'%'],'Fontsize',12);
set(gca,'YTick',1:5)
set(gca,'YTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
set(gca,'XTick',1:5)
set(gca,'XTickLabel',{'类别1' '类别2' '类别3' '类别4' '类别5'})
%% ===========================绘制ROC和AUC曲线======================
figure(1)
plotroc(output_train,Score1')
title('训练集ROC和AUC');
figure(2)
plotroc(output_test,Score')
title('测试集ROC和AUC');
%% ============================好看的混淆矩阵========================
%---------训练集---------
figure()
label = {'类别1' '类别2' '类别3' '类别4' '类别5'};
mat = cfmatrix(train_label,pre_Y1);
maxcolor = [191,54,12];   % 最大值颜色
mincolor = [255,255,255]; % 最小值颜色
% 绘制坐标轴
m = length(mat);
imagesc(1:m,1:m,mat)
xticks(1:m)
xlabel('预测类别','fontsize',10.5)
xticklabels(label)
yticks(1:m)
ylabel('实际类别','fontsize',10.5)
yticklabels(label)
% 构造渐变色
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];
colormap(mymap)
colorbar()
% 色块填充数字
for i = 1:m
    for j = 1:m
        text(i,j,num2str(mat(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',10);
    end
end
% 图像坐标轴等宽
ax = gca;
ax.FontName = 'SimHei';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
set(0,'defaultTextFontName', 'TimesSimSun'); %文字
axis square
title('训练集混淆矩阵')
%---------测试集----------
figure()
label = {'类别1' '类别2' '类别3' '类别4' '类别5'};
mat = cfmatrix(test_label,pre_Y);
maxcolor = [120,60,150];   % 最大值颜色
mincolor = [255,255,255]; % 最小值颜色
% 绘制坐标轴
m = length(mat);
imagesc(1:m,1:m,mat)
xticks(1:m)
xlabel('预测类别','fontsize',10.5)
xticklabels(label)
yticks(1:m)
ylabel('实际类别','fontsize',10.5)
yticklabels(label)
% 构造渐变色
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];
colormap(mymap)
colorbar()
% 色块填充数字
for i = 1:m
    for j = 1:m
        text(i,j,num2str(mat(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',10);
    end
end
% 图像坐标轴等宽
ax = gca;
ax.FontName = 'SimHei';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
set(0,'defaultTextFontName', 'TimesSimSun'); %文字
axis square
title('测试集混淆矩阵')



