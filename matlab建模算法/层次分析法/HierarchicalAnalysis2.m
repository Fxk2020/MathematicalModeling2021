% �������ߵĽű�
disp('������A��Z��Ȩ�ؾ���');
A=input('A-Z=');
disp('������B��A��Ȩ�ؾ���');
B=input('B-A=');
disp('�����Ǹ��־��ߵı��أ�');
w=B*A;
disp(w);
[M,I]=max(w);
%disp('����ѡ��ڼ��ַ���');
%disp(I)
HierarchicalAnalysis3(I)