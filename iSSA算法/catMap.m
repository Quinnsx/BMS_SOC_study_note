%% cat ”≥…‰
function Xout = catMap(dim)
    a = 1;b = 1;
    x1 = zeros(1,dim);
    y1 = zeros(1,dim);
    x = rand(dim);
    y = rand(dim);
    N = 1;
    for i = 1:dim
        x1(i) = mode(x(i) + b.*y(i),N);
        y1(i) = mode(a*x(i) + a.*b.*y(i),N);
    end
    Xout = (x1 - min(x1))./(max(x1) - min(x1));
end