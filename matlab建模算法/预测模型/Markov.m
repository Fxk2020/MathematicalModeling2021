function [] = Markov()
clc,clear;
A=input('Ŀǰ���г�ռ����Ϊ��');
B=input('�г���ת�����ʾ���Ϊ��');
C=A*B;
disp('��һ���г��ı仯���г�ռ����Ϊ��')
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
disp('���г��ȶ�������µ��г�ռ����Ϊ')
disp(X);
end

