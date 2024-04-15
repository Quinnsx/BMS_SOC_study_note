%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 主程序运行开始 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 程序初始阶段
clear all 
close all
clc

%% 初始化各算法参数
SearchAgents_no=30; %种群数量大小
N=SearchAgents_no;
pop_num=SearchAgents_no;
M=5;
Function_name='F7'; %选择测试函数
Max_iteration=300; %最大迭代次数
Max_iter=Max_iteration;

%% 初始化算法边界
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); 
 X = initialization_3(N, dim, ub, lb);

%% 多种智能优化算法 
[Best_pos,Best_score,SSA_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);                         %1.原始麻雀算法    （SSA)
[Best_pos_ILL_SSA,Best_score_ILL_SSA,ILL_SSA_curve]=ISSA_2036(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); %2.改进麻雀算法    (ISSA)
[PSO_gBestScore,PSO_gBest,PSO_cg_curve]=PSO(N,Max_iteration,lb,ub,dim,fobj);                               %3.原始粒子群算法  （PSO)
[fMin_GWO,bestX_GWO,GWO_curve]=GWO(pop_num,Max_iter,lb,ub,dim,fobj);                                       %4.原始灰狼优化算法 (GWO) 
[fMin_WOA,bestX_WOA,WOA_curve]=WOA(pop_num,Max_iter,lb,ub,dim,fobj);                                       %5.原始鲸鱼优化算法（WOA) 
% [ABest_scoreChimp,ABest_posChimp,Chimp_curve]=Chimp(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);         %6.原始黑猩猩算法  (Chimp)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可替换的各种优化算法 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[fMin_EWOA,bestX_EWOA,EWOA_curve]=E_WOA(pop_num,Max_iter,lb,ub,dim,fobj);%改进鲸鱼优化算法（EWOA) 
%[Best_pos_ISSA,Best_score_ISSA,ISSA_curve]=ISSA(M,Max_iteration,lb,ub,dim,fobj);%改进麻雀算法1(ISSA_1)
%[PSO_gBestScore,PSO_gBest,PSO_cg_curve]=MPSO(N,Max_iteration,lb,ub,dim,fobj);%改进粒子群算法（MPSO)
%[PSO_gBestScore,PSO_gBest,PSO_cg_curve]=TACPSO(N,Max_iteration,lb,ub,dim,fobj);%基于时变改进的粒子群算法（TACPSO)
%[fMin_IGWO,bestX_IGWO,IGWO_curve]=IGWO(pop_num,Max_iter,lb,ub,dim,fobj);%改进灰狼优化算法(IGWO)
%[HHO_gBestScore,HHO_gBest,HHO_cg_curve]=HHO(N,Max_iteration,lb,ub,dim,fobj);%原始哈里斯鹰算法（HHO)
%[ESCA_Best_score, ESCA_Best_pos, ESCA_Curve] = ESCA(X, N, Max_iteration, lb, ub, dim, fobj);%改进正余弦算法（ESCA)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 结束 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 绘制测试函数图像
figure
func_plot(Function_name);
title('搜索空间')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])

%% 绘制迭代曲线图像

figure;
t = 1:Max_iteration;
% semilogy(t, SSA_curve, 'blacko-',t,  ILL_SSA_curve, 'ro-',t, PSO_cg_curve, 'yo-',t,GWO_curve,'bO-',t,WOA_curve,'mo-', 'linewidth', 1.5, 'MarkerSize', 6, 'MarkerIndices', 1:25:Max_iteration);
% 假设t, SSA_curve等都已经定义
hLines = semilogy( t, SSA_curve, 'o-',t, ILL_SSA_curve, 'o-', t, PSO_cg_curve, 'o-', t, GWO_curve, 'o-', t, WOA_curve, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
set(hLines, 'MarkerIndices', 1:25:Max_iteration); % 假设Max_iteration已定义

% 定义每条线的颜色
% C1 = addcolor(1);
% C2 = addcolor(98);
% C3 = addcolor(135);
% C4 = addcolor(214);
% C5 = addcolor(215);

% 现在使用set函数设置每条线的样式
set(hLines(1), 'LineStyle', '--', 'Marker', 'd', 'LineWidth', 1, 'Color', [252,140,90]/255);
set(hLines(2), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 1, 'Color', [219,49,36]/255);
set(hLines(3), 'LineStyle', '--', 'Marker', '^', 'LineWidth', 1, 'Color', [144,190,224]/255);
set(hLines(4), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 1, 'Color', [75,116,178]/255);
set(hLines(5), 'LineStyle', '--', 'Marker', 'd', 'LineWidth', 1, 'Color',[255,223,146]/255);



% title(Function_name)
xlabel('迭代次数');
ylabel('适应度值');
axis fill
% grid on                                                        
box on
legend('SSA','ISSA','PSO','GWO','WOA');
set(gca,'fontname','宋体 ')
% ylim([-12569.5, 0]);

%% 输出各算法最优结果
display(['SSA最优值 : ', num2str(Best_score)]);
display(['ILL_SSA最优值 : ', num2str(Best_score_ILL_SSA)]);
display(['PSO最优值 : ', num2str(PSO_gBestScore)]);
% display(['Chimp最优值 : ', num2str(ABest_scoreChimp)]);
display(['GWO最优值 : ', num2str(fMin_GWO)]);
display(['WOA最优值 : ', num2str(fMin_WOA)]);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 主程序运行结束 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%