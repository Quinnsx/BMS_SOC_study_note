
function [Best_pos,Best_score,curve]=ISSA(pop,Max_iter,lb,ub,dim,fobj)

ST = 0.6;%预警值
PD = 0.7;%发现者的比列，剩下的是加入者
SD = 0.3;%意识到有危险麻雀的比重

PDNumber = round(pop*PD); %发现者数量
SDNumber = round(SD*pop);%意识到有危险麻雀数量
if(max(size(ub)) == 1)
    ub = ub.*ones(1,dim);
    lb = lb.*ones(1,dim);
end

%% 改进点1：circle映射初始化种群
X0=initializationcircle(pop,dim,ub,lb);
X = X0;

%%生成反向个体并选择
X_reverse = repmat(lb, pop, 1) + rand(pop, dim) .* (repmat(ub, pop, 1) - X);
%计算初始适应度值
fitness = zeros(1, pop);
fitness_reverse = zeros(1, pop);

% fitness = zeros(1,pop);
for i = 1:pop
    fitness(i) =  fobj(X(i,:));
    fitness_reverse(i) = fobj(X_reverse(i,:));
end

%% 选择适应度更好的个体
for i = 1:pop
    if fitness_reverse(i) < fitness(i)
        X(i,:) = X_reverse(i,:);
    end
end

[fitness, index]= sort(fitness);%排序
BestF = fitness(1);
WorstF = fitness(end);
GBestF = fitness(1);%全局最优适应度值
for i = 1:pop
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%全局最优位置
X_new = X;
for i = 1: Max_iter
    %% 改进点2：比例系数改进
    ST = 0.6;%预警值
    b = 0.7;%比例系数
    k = 0.1;%扰动因子
    r = b*(tan(-pi*i/(4*Max_iter) + pi/4))-k*rand();
    %防止r为负数或者大于1的数
    if r<0
        r=0.1;
    end
    if r>1
        r=1;
    end
    PDNumber = round(pop*r);
    SDNumber = round(pop*(1-r));


    BestF = fitness(1);
    WorstF = fitness(end);


    R2 = rand(1);
    for j = 1:PDNumber % 对每个发现者
        if(R2<ST)
            %% 改进点3：改进探索者位置更新公式
            X_new(j,:) = X(j,:).*(2/exp(4*i./(rand()*Max_iter)^2));
        else
            X_new(j,:) = X(j,:) + randn().*ones(1,dim);
        end
    end
    for j = PDNumber+1:pop
        %        if(j>(pop/2))
        if(j>(pop - PDNumber)/2 + PDNumber)
            X_new(j,:)= randn(1,dim).*exp((X(end,:) - X(j,:))/j^2);
        else
            %产生-1，1的随机数
            A = ones(1,dim);
            for a = 1:dim
                if(rand()>0.5)
                    A(a) = -1;
                end
            end
            AA = A'*inv(A*A');
            X_new(j,:)= X(1,:) + abs(X(j,:) - X(1,:)).*AA';
        end
    end
    Temp = randperm(pop);
    SDchooseIndex = Temp(1:SDNumber);
    for j = 1:SDNumber
        if(fitness(SDchooseIndex(j))>BestF)
            X_new(SDchooseIndex(j),:) = X(1,:) + randn().*abs(X(SDchooseIndex(j),:) - X(1,:));
        elseif(fitness(SDchooseIndex(j))== BestF)
            K = 2*rand() -1;
            X_new(SDchooseIndex(j),:) = X(SDchooseIndex(j),:) + K.*(abs( X(SDchooseIndex(j),:) - X(end,:))./(fitness(SDchooseIndex(j)) - fitness(end) + 10^-8));
        end
    end
    %边界控制
    for j = 1:pop
        for a = 1: dim
            if(X_new(j,a)>ub(a))
                X_new(j,a) =ub(a);
            end
            if(X_new(j,a)<lb(a))
                X_new(j,a) =lb(a);
            end
        end
    end
    %更新位置
    for j=1:pop
        fitness_new(j) = fobj(X_new(j,:));
    end
    for j = 1:pop
        if(fitness_new(j) < GBestF)
            GBestF = fitness_new(j);
            GBestX = X_new(j,:);
        end
    end
    %% 改进点4：tent扰动和柯西扰动
    Avgf = mean(fitness_new);
    x0 = rand;
    for j = 1:pop
        if fitness_new(j)<Avgf %柯西变异
            Temp = X_new(j,:);
            r = tan((rand() - 0.5)*pi);%柯西随机数
            Temp = Temp + r.*Temp;
            Temp(Temp>ub) = ub(Temp>ub);
            Temp(Temp<lb) = lb(Temp<lb);
            fitTemp = fobj(Temp);
            if fitTemp<fitness_new(j)
                fitness_new(j) = fitTemp;
                X_new(j,:) = Temp;
            end
        else%Tent扰动
            if x0<0.5
                tentV = 2*x0+rand/pop;
            else
                tentV = 2*(1-x0)+rand/pop;
            end
            Temp = X_new(j,:);
            Temp = Temp + tentV.*Temp;
            Temp(Temp>ub) = ub(Temp>ub);
            Temp(Temp<lb) = lb(Temp<lb);
            fitTemp = fobj(Temp);
            if fitTemp<fitness_new(j)
                fitness_new(j) = fitTemp;
                X_new(j,:) = Temp;
            end
            x0 = tentV;
        end
    end

    X = X_new;
    fitness = fitness_new;
    %排序更新
    [fitness, index]= sort(fitness);%排序
    BestF = fitness(1);
    WorstF = fitness(end);
    for j = 1:pop
        X(j,:) = X(index(j),:);
    end
    curve(i) = GBestF;
end
Best_pos =GBestX;
Best_score = curve(end);
end



