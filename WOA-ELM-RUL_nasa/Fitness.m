function mae_ = Fitness(x)
C = x(1);
tho = x(2);
global  Elm_Type Kernel_Type Pn_train Pn_test Tn_train Tn_test

P=Pn_train';Pt=Pn_test';T=Tn_train';Tt=Tn_test';
[Omega_train,OutputWeight] = elmtrain_kernel(P',T',Elm_Type,C,tho,Kernel_Type);
Omega_test = kernel_matrix(P,Kernel_Type, tho,Pt);
T_sim_train=(Omega_test' * OutputWeight)';  
mae_ = mean(abs(T_sim_train - Tt'));
