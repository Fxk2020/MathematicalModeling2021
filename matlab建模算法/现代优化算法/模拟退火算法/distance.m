function[fare]=distance(coord)
% 根 据 各 城 市 的 距 离 坐 标 求 相 互 之 间 的 距 离
% fare 为 各 城 市 的 距 离 ， coord 为 各 城 市 的 坐 标
[v,m]=size(coord);%m为城市的个数
fare=zeros(m);
for i=1:m%外层为行
for j=1:m%内层为列
fare(i,j)=(sum((coord(:,i)-coord(:,j)).^2))^0.5;
fare(j,i)=fare(i,j);%距离矩阵对称
end
end