% ��һ�������������
clc,clear
disp('��������߾���A(n��)');
A=input('A=');
[m,n]=size(A);
for i=1:n
    panduan = input('�����ǳɱ���(0)����Ч����(1)��');
    if panduan==0
        A(:,i)=Normalized_cost(A(:,i));
    end
    if panduan==1
        A(:,i)=Normalized_benefit(A(:,i));
    end
end
disp(A)
