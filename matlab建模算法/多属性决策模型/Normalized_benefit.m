% 用于效益属性的归一化处理
function f = Normalized_benefit(x)
    M = max(x);
    for i=1:length(x)
        x(i)=x(i)/M;
    end
    f = x;
end