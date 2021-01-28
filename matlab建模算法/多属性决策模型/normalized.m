% 归一化处理决策属性
clc,clear
disp('请输入决策矩阵A(n阶)');
A=input('A=');
[m,n]=size(A);
for i=1:n
    panduan = input('该列是成本型(0)还是效益型(1)？');
    if panduan==0
        A(:,i)=Normalized_cost(A(:,i));
    end
    if panduan==1
        A(:,i)=Normalized_benefit(A(:,i));
    end
end
disp(A)
