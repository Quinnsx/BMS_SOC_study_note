function [Omega_train OutputWeight] = elmtrain_kernel(P,T,Elm_Type,C,tho,Kernel_Type)

% 源码参考
%%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       APRIL 2004

%---------------------------------
% Input
%****** P            : 训练集输入
%****** T            : 训练集输出
%****** Elm_Type     : 回归0 or 分类1
%****** C            : 正则化系数
%****** tho          : 核参数
%****** Kernel_Type  : 核类型 

%----------------------------------

if Elm_Type  == 1
    T  = ind2vec(T);
end

n = size(T,2);

Omega_train = kernel_matrix(P',Kernel_Type, tho);
OutputWeight=((Omega_train+speye(n)/C)\(T')); 
% Y=(Omega_train * OutputWeight)';
