% ���ڳɱ����ԵĹ�һ������
function f = Normalized_cost(x)
    M = min(x);
    for i=1:length(x)
        x(i)=M/x(i);
    end
    f = x;
end