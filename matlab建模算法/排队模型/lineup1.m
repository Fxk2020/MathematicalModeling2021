%排队论的M/M/S模型
clear,clc;
%服务台的个数
% s=2;
s=3;
%单位时间(h)内单个服务台能处理的人数
% mu=4;
mu=24;
%单位时间（h）到达的人数
% lambda=3;
lambda=54;
[Lq,L,W,Wq]=MMs(s,mu,lambda);
fprintf('排队等待的平均人数为%5.2f人\n',Lq)
fprintf('系统内的平均人数为%5.2f人\n',L)
fprintf('平均逗留时间为%5.2f分钟\n',W*60)
fprintf('平均等待时间为%5.2f分钟\n',Wq*60)