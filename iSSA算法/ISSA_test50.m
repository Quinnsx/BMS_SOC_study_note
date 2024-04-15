%% 程序初始阶段
clear all 
close all
clc

%% 初始化各算法参数
SearchAgents_no=30; %种群数量大小
N=SearchAgents_no;
M=5;
Function_name='F7'; %选择测试函数
Max_iteration=300; %最大迭代次数
Max_iter=Max_iteration;
NumRuns = 300; % 算法运行次数
%% 初始化算法边界
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); 
 X = initialization_3(N, dim, ub, lb);
% 初始化结果存储数组
Best_scores_ILL_SSA = zeros(NumRuns, 1);

%% 执行算法50次
for i = 1:NumRuns
    [Best_pos_ILL_SSA_temp, Best_score_ILL_SSA_temp, ILL_SSA_curve] = ISSA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj);
    % [ Best_score_ILL_SSA_temp,Best_pos_ILL_SSA_temp, ILL_SSA_curve] = GWO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj);
    Best_scores_ILL_SSA(i) = Best_score_ILL_SSA_temp;
end

% 计算平均值和标准差
Avg_ILL_SSA = mean(Best_scores_ILL_SSA);
Std_ILL_SSA = std(Best_scores_ILL_SSA);
Var_ILL_SSA = var(Best_scores_ILL_SSA);
Min_ILL_SSA = min(Best_scores_ILL_SSA);
%% 输出结果
fprintf('ISSA平均最优值: %.10e\n', Avg_ILL_SSA);
fprintf('ISSA最优值的标准差: %.10e\n', Std_ILL_SSA);
fprintf('ISSA最优值的方差: %.10e\n', Var_ILL_SSA);
fprintf('ISSA最优值: %.10e\n', Min_ILL_SSA);

display(['ISSA平均最优值 : ', num2str(Avg_ILL_SSA)]);
display(['ISSA最优值的标准差 : ', num2str(Std_ILL_SSA)]);
display(['ISSA最优值的方差 : ', num2str(Var_ILL_SSA)]);
display(['ISSA最优值 : ', num2str(Min_ILL_SSA)]);
%% 绘制迭代曲线图像（如果有需要）
% 这里你需要确保你的ISSA函数返回一个迭代曲线，例如ILL_SSA_curve
figure;
t = 1:Max_iteration;
semilogy(t, ILL_SSA_curve, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('迭代次数');
ylabel('适应度值');
title('ISSA算法迭代曲线');
legend('ISSA');

%% 绘制最优值分布图像
% 绘制箱线图
figure;
boxplot(Best_scores_ILL_SSA, 'Labels', {'ISSA'});
xlabel('迭代次数');
ylabel('最优值');
title('ISSA算法最优值箱线图');