
function [Leader_score,Leader_pos,Convergence_curve]=WOA(SearchAgents_no,Max_iter,lb,ub,dim,fobj)


% initialize position vector and score for the leader
Leader_pos=zeros(1,dim);
Leader_score=inf; %change this to -inf for maximization problems

%种群规模  SearchAgents_no
%最大迭代次数 Max_iter
% lb;  %自变量下边界
% ub;   %自变量上边界
% dim;   %自变量维度
% fobj;   %适应度函数

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);  % 适应度变化曲线

t=0;% Loop counter

% Main loop
while t<Max_iter
    for i=1:size(Positions,1)
        
        % Return back the search agents that go beyond the boundaries of the search space
        % 判断上下限是否超过，并校正
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
        % Calculate objective function for each search agent
        fitness=fobj(Positions(i,:));
        % Update the leader
        if fitness<Leader_score % Change this to > for maximization problem
            Leader_score=fitness; % Update alpha 最优鲸鱼适应度
            Leader_pos=Positions(i,:);  % 最优鲸鱼位置
        end
        
    end
    
    a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)  公式11
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/Max_iter);
    
    % Update the Position of search agents 
    for i=1:size(Positions,1)
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        
        A=2*a*r1-a;  % Eq. (2.3) in the paper   公式11
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        
        p = rand();        % p in Eq. (2.6)
        
        for j=1:size(Positions,2)
            
            if p<0.5      % 公式15判别
                if abs(A)>=1  % 当收敛因子 | A| ≥ 1 时，采用随机更新自身位方式进行全局搜索
                    rand_leader_index = floor(SearchAgents_no*rand()+1);  % 随机鲸鱼个体
                    X_rand = Positions(rand_leader_index, :);       % 随机鲸鱼个体位置
                    D_X_rand=abs(C*X_rand(j)-Positions(i,j)); % Eq. (2.7)  公式17
                    Positions(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)  公式16
                    
                elseif abs(A)<1
                    D_Leader=abs(C*Leader_pos(j)-Positions(i,j)); % Eq. (2.1)  公式15中的上部分公式中的D
                    Positions(i,j)=Leader_pos(j)-A*D_Leader;      % Eq. (2.2)   公式15中的上部分公式
                end
                
            elseif p>=0.5
              
                distance2Leader=abs(Leader_pos(j)-Positions(i,j)); % Leader_pos;  % 最优鲸鱼位置
                % Eq. (2.5)
                Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j); % 公式15中的上部分公式
                
            end
            
        end
    end
    t=t+1;
    Convergence_curve(t)=Leader_score;
    % figure(1)
    % plot(Convergence_curve,'r-o','MarkerIndices',1:3:100,'Markersize',4,'LineWidth', 1.0);
    % legend('迭代曲线');
    % xlabel('迭代次数');
    % set(gcf, 'Color', [1,1,1])
    % set(gca,'linewidth',1,'fontsize',12);
    % box on
    % grid on
    [t Leader_score];
end



