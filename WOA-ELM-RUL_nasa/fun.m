function y = fun(x)
pm=reshape(x,1,length(x));
global  indim hiddennum outdim D Pn_train Pn_test Tn_train Tn_test
Ptrain=Pn_train;
Ptest=Pn_test;
Ttrain=Tn_train;
Ttest=Tn_test;

    for j=1:hiddennum
        x2iw(j,:)=pm(((j-1)*indim+1):j*indim);
    end
    for k=1:outdim
        x2lw(k,:)=pm((indim*hiddennum+1):(indim*hiddennum+hiddennum));
    end
    x2b=pm(((indim+1)*hiddennum+1):D);
    x2b1=x2b(1:hiddennum).';
    x2b2=x2b(hiddennum+1:hiddennum+outdim).';

    IW1=x2iw ;
    IW2=x2lw;  
    b1=x2b1;
    b2=x2b2;
    
    P1=Ptrain;P2=Ptest;T1=Ttrain;T2=Ttest;
    [LW,TF,TYPE] = elmtrain(P1,T1,hiddennum,'sig',0,IW1,b1);
    T_sim = elmpredict(P2,IW1,b1,LW,TF,TYPE);
    err=T_sim-T2;
    y=mse(err);    
end