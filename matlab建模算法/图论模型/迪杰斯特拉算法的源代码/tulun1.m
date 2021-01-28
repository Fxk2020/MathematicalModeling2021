% weight=[0 2 8 1 Inf Inf Inf Inf Inf Inf Inf
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
clear,clc;
% weight=[0 2 4 Inf Inf Inf Inf
%     Inf 0 Inf 3 3 1 Inf
%     Inf Inf 0 2 3 1 Inf
%     Inf Inf Inf 0 Inf Inf 1
%     Inf Inf Inf Inf 0 Inf 3
%     Inf Inf Inf Inf Inf 0 4
%     Inf Inf Inf Inf Inf Inf 0];
weight=input('请输入邻接矩阵：');
start=input('输入开始节点');
endD=input('输入终止节点');
[dis,path]=dijkstra(weight,start,endD);
disp(dis);
disp(path);