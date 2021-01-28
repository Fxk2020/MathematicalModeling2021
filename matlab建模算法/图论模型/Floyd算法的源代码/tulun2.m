% a=[0 2 8 1 Inf Inf Inf Inf Inf Inf Inf
% 2 0 6 Inf 1 Inf Inf Inf Inf Inf Inf
% 8 6 0 7 5 1 2 Inf Inf Inf Inf
% 1 Inf 7 0 Inf Inf 9 Inf Inf Inf Inf
% Inf 1 5 Inf 0 3 Inf 2 9 Inf Inf
% Inf Inf 1 Inf 3 0 4 Inf 6 Inf Inf
% Inf Inf 2 9 Inf 4 0 Inf 3 1 Inf
% Inf Inf Inf Inf 2 Inf Inf 0 7 Inf 9
% Inf Inf Inf Inf 9 6 3 7 0 1 2
% Inf Inf Inf Inf Inf Inf 1 Inf 1 0 4
% Inf Inf Inf Inf Inf Inf Inf 9 2 4 0];

% a=[0 50 Inf 40 25 10
%     50 0 15 20 Inf 25
%     Inf 15 0 10 20 Inf
%     40 20 10 0 10 25
%     25 Inf 20 10 0 55
%     10 25 Inf 25 55 0];
% 使用folyd算法
clear,clc;
% a=[Inf 12 Inf Inf Inf 16 14
%     12 Inf 10 Inf Inf 7 Inf
%     Inf 10 Inf 3 5 6 Inf
%     Inf Inf 3 Inf 4 Inf Inf
%     Inf Inf 5 4 Inf 2 8
%     16 7 6 Inf 2 Inf 9
%     14 Inf Inf Inf 8 9 Inf];
% a= [0     8   Inf   Inf   Inf   Inf     7     8   Inf   Inf   Inf;
%    Inf     0     3   Inf   Inf   Inf   Inf   Inf   Inf   Inf   Inf;
%    Inf   Inf     0     5     6   Inf     5   Inf   Inf   Inf   Inf;
%    Inf   Inf   Inf     0     1   Inf   Inf   Inf   Inf   Inf    12;
%    Inf   Inf     6   Inf     0     2   Inf   Inf   Inf   Inf    10;
%    Inf   Inf   Inf   Inf     2     0     9   Inf     3   Inf   Inf;
%    Inf   Inf   Inf   Inf   Inf     9     0   Inf   Inf   Inf   Inf;
%      8   Inf   Inf   Inf   Inf   Inf   Inf     0     9   Inf   Inf;
%    Inf   Inf   Inf   Inf     7   Inf   Inf     9     0     2   Inf;
%    Inf   Inf   Inf   Inf   Inf   Inf   Inf   Inf     2     0     2;
%    Inf   Inf   Inf   Inf    10   Inf   Inf   Inf   Inf   Inf     0;];
a = input('请输入各个顶点之间的邻接矩阵：');
[D,path]=floyd(a);
disp('费用矩阵为：');
disp(D);
disp('最短路径为');
disp(path);