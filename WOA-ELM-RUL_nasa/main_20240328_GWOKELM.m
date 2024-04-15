%% I. 清空环境变量
clear 
close all;
%% II. 导入数据 
% 设置文件路径
filePath = ['D:\embed_work\python_project\issa-lightgbm\output\PC2\cell3_pca.csv'];
% filePath2 = 'D:\embed_work\python_project\issa-lightgbm\output\PC2\cell3_pca.csv';


%% 2. 划分数据集
global  Kernel_Type Elm_Type Kernel_Type Pn_train Pn_test Tn_train Tn_test

dataTable = readtable(filePath);

inputColumns = dataTable(:, {'PC1','PC2'});
outputColumn = dataTable(:, 'SOH');

inputArray = table2array(inputColumns);
outputArray = table2array(outputColumn);

n=size(inputArray,1);
m=round(n*0.5);   %前70%训练，对最后30%进行预测
% m =39
train_attributes=inputArray(1:m,:);
train_strength =outputArray(1:m,:);
test_attributes= inputArray(m+1:end,:);
test_strength = outputArray(m+1:end,:);


train_attributes=train_attributes';
train_strength = train_strength';
test_attributes = test_attributes';
test_strength = test_strength';

P_train  = train_attributes;
T_train = train_strength; 
P_test  = test_attributes;
T_test =  test_strength; 

M = size(P_train, 2);
N = size(P_test, 2);


%% III. 数据归一化
% 1. 训练集
[Pn_train,inputps] = mapminmax(P_train);
% Pn_train=Pn_train';
Pn_test = mapminmax('apply',P_test,inputps);

% 2. 测试集
[Tn_train,outputps] = mapminmax(T_train);
% Tn_train=Tn_train';
Tn_test = mapminmax('apply',T_test,outputps);

%% ELM基本参数设置
indim=size(P_train,1);%输入节点数
hiddennum=indim+1;%中间节点数
outdim=1;%输出节点数
        
%% 鲸鱼算法的基本参数初始化
% dim = (indim+1)*hiddennum+(hiddennum+1)*outdim;   
% D=dim;
% Max_iteration=110;   % 进化次数  
% pop=30;  %种群规模
% lb = -10*ones(1,dim);
% ub =20*ones(1,dim);
% fobj = @(x) fun(x);
% [Leader_score,Leader_pos,WOA_curve]=WOA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化
% Bestf=Leader_pos; %最优个体 
% 
% clc
% pm=Bestf;
% pm=reshape(pm,1,length(pm));
% for j=1:hiddennum
%    x2iw(j,:)=pm(((j-1)*indim+1):j*indim);
% end
% for k=1:outdim
%    x2lw(k,:)=pm((indim*hiddennum+1):(indim*hiddennum+hiddennum));
% end
% x2b=pm(((indim+1)*hiddennum+1):D);
% x2b1=x2b(1:hiddennum).';
% x2b2=x2b(hiddennum+1:hiddennum+outdim).';
% IW1=x2iw ;
% IW2=x2lw;  
% b1=x2b1;
% b2=x2b2;
% % disp(['best_IW1:',num2str(IW1)])
% % disp(['best_b1:',num2str(b1)])
% [LW,TF,TYPE] = elmtrain(Pn_train,Tn_train,hiddennum,'sig',0,IW1,b1);
% tn_sim = elmpredict(Pn_test,IW1,b1,LW,TF,TYPE);
% WOA_ELM_T_sim0 = mapminmax('reverse',tn_sim,outputps);
% disp('程序优化过程运行结束');
% 
% %%
% WOA_ELM_error=WOA_ELM_T_sim0-T_test;

%% KELM参数
Elm_Type     = 0;            % 0 是回归拟合  1是分类
Kernel_Type  =  'RBF_kernel';
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
fun = @Fitness; % 适应度函数  
popmin= [eps eps];    % 将边界值改为向量
popmax= [1.0e+10 1.0e+10];    % 将边界值改为向量
totalnum = 2;

%% 灰狼算法参数初始化
lb=popmin;
ub=popmax;
dim=totalnum;
fobj = @Fitness;
popsize = 30;
iter_max = 100;

[gBestFitness,gBest,IterCurve]=GWO(popsize,iter_max,lb,ub,dim,fobj)

%% 迭代曲线
plot(IterCurve,'Color',[021 151 165]/255,'LineWidth',3);
title('最优个体适应度','fontsize',12);
xlabel('进化代数','fontsize',12);ylabel('适应度','fontsize',12);
% Best_Pos

%% 最优KELM训练
C = gBest(1);
tho = gBest(2);


% KELM创建/训练
[Omega_train OutputWeight] = elmtrain_kernel(Pn_train,Tn_train,Elm_Type,C,tho,Kernel_Type);
GWO_KELM_sim_train=(Omega_train * OutputWeight)';   % 训练集预测输出

% KELM预测
tn_sim = elmpredict_kernel(Pn_train,Pn_test,OutputWeight,Kernel_Type, tho,Elm_Type);
GWO_KELM_T_sim0 = mapminmax('reverse',tn_sim,outputps);
GWO_KELM_error = GWO_KELM_T_sim0 -T_test
%%
%% IV. ELM创建/训练
[IW,B,LW,TF,TYPE] = ELM_train(Pn_train,Tn_train,hiddennum,'sig',0);   
ELM_tn_sim = ELM_predict(Pn_test,IW,B,LW,TF,TYPE);
ELM_T_sim = mapminmax('reverse',ELM_tn_sim,outputps);
ELM_error=ELM_T_sim-T_test;


%% VII. 绘图
%% 结果图
figure(2)
plot(T_test,'k-','LineWidth',1.65);
hold on
plot(GWO_KELM_T_sim0 ,'r--','LineWidth', 1.5);
plot(ELM_T_sim ,'b-.','LineWidth', 1.5);

legend('实际值','WOA-ELM','ELM');
legend('boxoff');
xlabel('B0006-cycle');
ylabel('Capacity/Ah');
set(gcf, 'Color', [1,1,1])
set(gca,'linewidth',1,'fontsize',12);
hold on;
box on;
grid on;

%% 电池容量误差曲线图
figure(3)
plot(GWO_KELM_error,'r-o','MarkerIndices',1:2:30,'Markersize',4,'LineWidth', 1.0);
hold on;
plot(ELM_error,'b-p','MarkerIndices',1:2:30,'Markersize',4,'LineWidth', 1.0);
xlabel('B0006-cycle');
ylabel('误差');
legend('WOA-ELM','ELM');

%%误差计算及输出
%%  均方根误差 RMSE
error1 = sqrt(sum((GWO_KELM_T_sim0 - T_test).^2)./M);
error2 = sqrt(sum((ELM_T_sim - T_test).^2)./N);
% error3 = sqrt(sum((Chimp_ELM_T_sim0 - T_test).^2)./N);
%%
%决定系数
R1 = 1 - norm(T_test - GWO_KELM_T_sim0)^2 / norm(T_test - mean(T_test))^2;
R2 = 1 - norm(T_test -  ELM_T_sim)^2 / norm(T_test -  mean(T_test ))^2;
% R3 = 1 - norm(T_test -  Chimp_ELM_T_sim0)^2 / norm(T_test -  mean(T_test ))^2;

%%
%均方误差 MSE
mse1 = sum((GWO_KELM_T_sim0 - T_test).^2)./M;
mse2 = sum((ELM_T_sim - T_test).^2)./N;
% mse3 = sum((Chimp_ELM_T_sim0 - T_test).^2)./N;
%% 平均绝对误差MAE
MAE1 = mean(abs(T_test - GWO_KELM_T_sim0));
MAE2 = mean(abs(T_test - ELM_T_sim));
% MAE3 = mean(abs(T_test - Chimp_ELM_T_sim0));
%% 平均绝对百分比误差MAPE
MAPE1 = mean(abs((T_test - GWO_KELM_T_sim0)./T_test));
MAPE2 = mean(abs((T_test - ELM_T_sim)./T_test));
% MAPE3 = mean(abs((T_test - Chimp_ELM_T_sim0)./T_test));
%% 打印出评价指标
disp(['-----------------------误差计算--------------------------'])
disp(['WOA评价结果如下所示：'])
disp(['WOA平均绝对误差MAE为：',num2str(MAE1)])
disp(['WOA均方误差MSE为：       ',num2str(mse1)])
disp(['WOA均方根误差RMSEP为：  ',num2str(error1)])
disp(['WOA决定系数R^2为：  ',num2str(R1)])
% disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
disp(['WOA平均绝对百分比误差MAPE为：  ',num2str(MAPE1)])

% disp(['-----------------------误差计算--------------------------'])
% disp(['Chimp评价结果如下所示：'])
% disp(['Chimp平均绝对误差MAE为：',num2str(MAE3)])
% disp(['Chimp均方误差MSE为：       ',num2str(mse3)])
% disp(['Chimp均方根误差RMSEP为：  ',num2str(error3)])
% disp(['Chimp决定系数R^2为：  ',num2str(R3)])
% % disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
% disp(['Chimp平均绝对百分比误差MAPE为：  ',num2str(MAPE3)])
% grid


% 
% % 假设这些变量都是相同长度的 1xN 的向量
% ELM_T_sim1 = [ ... ]; % 您的数据
% ELM_T_sim2 = [ ... ];
% ELM_T_sim3 = [ ... ];
% ELM_T_sim4 = [ ... ];
% 
% % 将它们合并为一个矩阵
% dataToSave = [ELM_T_sim',WOA_ELM_T_sim0'];

% 使用 csvwrite
% csvwrite('cell1_result.csv', dataToSave);
