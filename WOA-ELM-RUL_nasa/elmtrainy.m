% ELM 训练网络
function [IW,B,LW,TF,TYPE] = elmtrainy(P,T,N,TF,TYPE)

if nargin < 2 
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 3  
    N = size(P,2);
end
if nargin < 4 
    TF = 'sig';
end
if nargin < 5 
    TYPE = 0;
end   
if size(P,2) ~= size(T,2)  
    error('ELM:Arguments','The columns of P and T must be same.');
end

[R,Q] = size(P); %

if TYPE  == 1
    T  = ind2vec(T);
end
[S,Q] = size(T);

IW = rand(N,R) * 2 - 1;
B = rand(N,1);
BiasMatrix = repmat(B,1,Q);

tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

LW = pinv(H') * T';


%  相关注释
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax 语法
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% Description 描述
% Input
% P   - Input Matrix of Training Set  (R*Q)  训练输入样本
% T   - Output Matrix of Training Set (S*Q) 训练输出样本
% N   - Number of Hidden Neurons (default = Q) 隐含层节点数
% TF  - Transfer Function: 传递函数，转化函数
%       'sig' for Sigmoidal function (default) S型函数
%       'sin' for Sine function 正弦函数
%       'hardlim' for Hardlim function 硬限制型传递函数
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R) 输入权值
% B   - Bias Matrix  (N*1) 偏差
% LW  - Layer Weight Matrix (N*S)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMPREDICT
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $