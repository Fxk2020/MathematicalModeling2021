% 用于成本属性的归一化处理
function f = Normalized_cost(x)
    M = min(x);
    for i=1:length(x)
        x(i)=M/x(i);
    end
    f = x;
end