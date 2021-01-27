% 做出决策的脚本
disp('请输入A到Z的权重矩阵：');
A=input('A-Z=');
disp('请输入B到A的权重矩阵：');
B=input('B-A=');
disp('下面是各种决策的比重：');
w=B*A;
disp(w);
[M,I]=max(w);
%disp('所以选择第几种方案');
%disp(I)
HierarchicalAnalysis3(I)