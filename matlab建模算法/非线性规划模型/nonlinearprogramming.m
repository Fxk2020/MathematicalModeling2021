[x,y]=fmincon('nonlinear_f',rand(3,1),[],[],[],[],zeros(3,1),[],'nonlinear_nonlcon');
disp('x��ֵΪ��')
disp(x);
disp('�õ�������ֵΪ��')
disp(y);