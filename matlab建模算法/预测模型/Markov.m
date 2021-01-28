function [] = Markov()
clc,clear;
A=input('目前的市场占有率为：');
B=input('市场的转换概率矩阵为：');
C=A*B;
disp('第一次市场的变化后市场占有率为：')
disp(C);
X = C;
T = true;
while T
    Y = X*B;
    if X == Y
        T = false;
    else
        X = X*B;
    end
end
disp('在市场稳定的情况下的市场占有率为')
disp(X);
end

