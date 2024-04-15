function Y = elmpredict_kernel(P,P_test,OutputWeight,Kernel_type, tho,Elm_Type)

% 源码参考
%%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       APRIL 2004

%----------------------
% Input
%****** P            : 训练集输入
%****** P_testT      : 测试集输入
%****** OutputWeight : 输出权值
%****** tho          : 核参数
%****** Kernel_Type  : 核类型 

%----------------------

Omega_test = kernel_matrix(P',Kernel_type, tho,P_test');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data
Y = TY;

if Elm_Type == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); 
end