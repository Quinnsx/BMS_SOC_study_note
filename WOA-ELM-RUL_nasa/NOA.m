% % 星鸦优化算法
% 微信公众号/知乎/CSDN:KAU的云实验台


function [Best_score,Best_NC,Convergence_curve]=NOA(SearchAgents_no,Max_iter,ub,lb,dim,fobj)

lb=ones(1,dim).*(lb);                              % Lower limit for variables
ub=ones(1,dim).*(ub);                              % Upper limit for variables


%% 初始参数定义
Best_NC=zeros(1,dim); % A vector to include the best-so-far Nutcracker(Solution) 最佳星鸦
Best_score=inf; % A Scalar variable to include the best-so-far score 最佳适应度
LFit=inf*(ones(SearchAgents_no,1)); % A vector to include the local-best position for each Nutcracker 种群 适应度
RP=zeros(2,dim); %%  2-D matrix to include two reference points of each Nutcracker 参考点
Convergence_curve=zeros(1,Max_iter);

%% 控制参数
Alpha=0.05; %% The percent of attempts at avoiding local optima
Pa2=0.2; %% The probability of exchanging between the cache-search stage and the recovery stage
Prb=0.2;    % The percentage of exploration other regions within the search space.

%% 种群初始化
Positions=initialization(SearchAgents_no,dim,ub,lb); %Initialize the positions of search agents
Lbest=Positions; %% Set the local best for each Nutcracker as its current position at the beginning.
t=0; %% Function evaluation counter

for i=1:SearchAgents_no

    NC_Fit(i)=fobj(Positions(i,:));

    LFit(i)=NC_Fit(i); %% Set the local best score for the ith Nutcracker as its current score.

    % Update the best-so-far solution
    if NC_Fit(i)<Best_score % Change this to > for maximization problem
        Best_score=NC_Fit(i); % Update the best-so-far score
        Best_NC=Positions(i,:); % Update te best-so-far solution
    end
end


%% 迭代
while t<Max_iter

    RL=0.05*levy(SearchAgents_no,dim,1.5);   %Levy random number vector 对应公众号文章中的勘探1中的y
    l=rand*(1-t/Max_iter);   %  Parameter in Eq. (3) 对应公众号文章中的开发1中的随机线性递减因子I

    % 对应公众号文章中的寻找储藏区与找回策略中的a
    if rand<rand
        a=(t/Max_iter)^(2*1/t);
    else
        a=(1-(t/Max_iter))^(2*(t/Max_iter));
    end

    if rand<rand % 随机选择进行第一个阶段或者是第二个阶段  每次迭代仅运行一个阶段的搜索

        % 小于就是第一个阶段——对应觅食与存储策略
        mo= mean(Positions);
        for i=1:SearchAgents_no

            % 对应公众号文章中的勘探1中的u
            if rand<rand
                mu=rand;
            elseif rand<rand
                mu=(randn);
            else
                mu=(RL(1,1));
            end

            cv=randi(SearchAgents_no);%% An index selected randomly between 1 and SearchAgents_no
            cv1=randi(SearchAgents_no); %% An index selected randomly between 1 and SearchAgents_no
            Pa1=((Max_iter-t)/Max_iter);
            if rand<Pa1
                % 觅食阶段——勘探1
                cv2=randi(SearchAgents_no);
                r2=rand;
                for j=1:size(Positions,2)
                    if t<Max_iter/2
                        if rand>rand
                            Positions(i,j)=(mo(j))+RL(i,j)*(Positions(cv,j)-Positions(cv1,j))+mu*(rand<5)*(r2*r2*ub(j)-lb(j));      % Eq. (1)
                        end
                    else
                        if rand>rand
                            Positions(i,j)=Positions(cv2,j)+mu*(Positions(cv,j)-Positions(cv1,j))+mu*(rand<Alpha)*(r2*r2*ub(j)-lb(j)); % Eq. (1)
                        end
                    end
                end
            else  
                % 存储阶段——开发1
                mu=rand;
                if rand<rand
                    r1=rand;
                    for j=1:size(Positions,2)
                        Positions(i,j)=((Positions(i,j)))+mu*abs(RL(i,j))*(Best_NC(j)-Positions(i,j))+(r1)*(Positions(cv,j)-Positions(cv1,j));      % Eq. (3)
                    end
                elseif rand<rand
                    for j=1:size(Positions,2)
                        if rand>rand
                            Positions(i,j)=Best_NC(j)+mu*(Positions(cv,j)-Positions(cv1,j)); % Eq. (3)
                        end
                    end
                else
                    for j=1:size(Positions,2)
                        Positions(i,j)=(Best_NC(j)*abs(l));      % Eq. (3)
                    end
                end
            end
            % 范围规约
            if rand<rand
                for j=1:size(Positions,2)
                    if  Positions(i,j)>ub(j)
                        Positions(i,j)=lb(j)+rand*(ub(j)-lb(j));
                    elseif  Positions(i,j)<lb(j)
                        Positions(i,j)=lb(j)+rand*(ub(j)-lb(j));
                    end
                end
            else
                Positions(i,:) = min(max(Positions(i,:),lb),ub);
            end

            NC_Fit(i)=fobj(Positions(i,:));

            % 贪婪策略更新
            if NC_Fit(i)<LFit(i) % Change this to > for maximization problem
                LFit(i)=NC_Fit(i); % Update the local best fitness
                Lbest(i,:)=Positions(i,:); % Update the local best position
            else
                NC_Fit(i)=LFit(i);
                Positions(i,:)=Lbest(i,:);
            end
            % Update the best-so-far solution
            if NC_Fit(i)<Best_score % Change this to > for maximization problem
                Best_score=NC_Fit(i); % Update best-so-far fitness
                Best_NC=Positions(i,:); % Update best-so-far position
            end
            t=t+1;
            if t>Max_iter
                break;
            end
            Convergence_curve(t)=Best_score;
        end
    else 
    % 寻找储藏区与找回食物策略
        % 首先计算参考点
        for i=1:SearchAgents_no
            ang=pi*rand;
            cv=randi(SearchAgents_no);
            cv1=randi(SearchAgents_no);
            for j=1:size(Positions,2)
                for j1=1:2
                    if j1==1
                        %% Compute the first reference point for the ith Nutcraker using Eq. (9)
                        if ang~=pi/2
                            RP(j1,j)=Positions(i,j)+ (a*cos(ang)*(Positions(cv,j)-Positions(cv1,j)));
                        else
                            RP(j1,j)=Positions(i,j)+ a*cos(ang)*(Positions(cv,j)-Positions(cv1,j))+a*RP(randi(2),j);
                        end
                    else
                        %% Compute the second reference point for the ith Nutcraker using Eq. (10)
                        if ang~=pi/2
                            RP(j1,j)=Positions(i,j)+ (a*cos(ang)*((ub(j)-lb(j))+lb(j)))*(rand<Prb);
                        else
                            RP(j1,j)=Positions(i,j)+ (a*cos(ang)*((ub(j)-lb(j))*rand+lb(j))+a*RP(randi(2),j))*(rand<Prb);
                        end
                    end
                end
            end
            % 参考点的范围规约
            if rand<rand
                for j=1:size(Positions,2)
                    if  RP(2,j)>ub(j)
                        RP(2,j)=lb(j)+rand*(ub(j)-lb(j));
                    elseif  RP(2,j)<lb(j)
                        RP(2,j)=lb(j)+rand*(ub(j)-lb(j));
                    end
                end
            else
                RP(2,:) = min(max(RP(2,:),lb),ub);
            end
            if rand<rand
                for j=1:size(Positions,2)
                    if  RP(1,j)>ub(j)
                        RP(1,j)=lb(j)+rand*(ub(j)-lb(j));
                    elseif  RP(1,j)<lb(j)
                        RP(1,j)=lb(j)+rand*(ub(j)-lb(j));
                    end
                end
            else
                RP(1,:) = min(max(RP(1,:),lb),ub);
            end

            % 寻找储藏区阶段——勘探2
            if (rand<Pa2) %% Exploitation stage 2: Recovery stage
                cv=randi(SearchAgents_no);
                if rand<rand
                    for j=1:size(Positions,2)
                        if rand>rand
                            Positions(i,j)=Positions(i,j)+rand*(Best_NC(j)-Positions(i,j))+rand*(RP(1,j)-Positions(cv,j));  %% Eq. (13)
                        end
                    end
                else
                    for j=1:size(Positions,2)
                        if rand>rand
                            Positions(i,j)=Positions(i,j)+rand*(Best_NC(j)-Positions(i,j))+rand*(RP(2,j)-Positions(cv,j)); %% Eq. (15)
                        end
                    end
                end
                % 范围规约
                if rand<rand
                    for j=1:size(Positions,2)
                        if  Positions(i,j)>ub(j)
                            Positions(i,j)=lb(j)+rand*(ub(j)-lb(j));
                        elseif  Positions(i,j)<lb(j)
                            Positions(i,j)=lb(j)+rand*(ub(j)-lb(j));
                        end
                    end
                else
                    Positions(i,:) = min(max(Positions(i,:),lb),ub);
                end
                % Calculate objective function for each search agent

                NC_Fit(i)=fobj(Positions(i,:));

                % Update the local best
                if NC_Fit(i)<LFit(i) % Change this to > for maximization problem
                    LFit(i)=NC_Fit(i);
                    Lbest(i,:)=Positions(i,:);
                else
                    NC_Fit(i)=LFit(i);
                    Positions(i,:)=Lbest(i,:);
                end
                % Update the best-so-far solution
                if NC_Fit(i)<Best_score % Change this to > for maximization problem
                    Best_score=NC_Fit(i); % Update best-so-far fitness
                    Best_NC=Positions(i,:); % Update best-so-far position
                end
                t=t+1;
                Convergence_curve(t)=Best_score;
                if t>Max_iter
                    break;
                end
            else
                % 找回食物——开发2

                NC_Fit1=fobj(RP(1,:));

                %%-------Evaluations-----------

                NC_Fit2=fobj(RP(2,:));

                %%%%%----------- Applying Eq. (17) to trade-off between the exploration behaviors---------%%%%
                if NC_Fit2<NC_Fit1 && NC_Fit2<NC_Fit(i)
                    Positions(i,:)=RP(2,:);
                    NC_Fit(i)=NC_Fit2;
                elseif NC_Fit1<NC_Fit2 && NC_Fit1<NC_Fit(i)
                    Positions(i,:)=RP(1,:);
                    NC_Fit(i)=NC_Fit1;
                end
                % Update the local best
                if NC_Fit(i)<LFit(i) % Change this to > for maximization problem
                    LFit(i)=NC_Fit(i);
                    Lbest(i,:)=Positions(i,:);
                else
                    NC_Fit(i)=LFit(i);
                    Positions(i,:)=Lbest(i,:);
                end
                t=t+1;
                % Update the best-so-far solution
                if NC_Fit(i)<Best_score % Change this to > for maximization problem
                    Best_score=NC_Fit(i);
                    Best_NC=Positions(i,:);
                end
                Convergence_curve(t)=Best_score;
                if t>Max_iter
                    break;
                end
            end
        end
    end
end
end
