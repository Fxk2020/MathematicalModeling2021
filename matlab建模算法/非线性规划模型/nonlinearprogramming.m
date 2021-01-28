[x,y]=fmincon('nonlinear_f',rand(3,1),[],[],[],[],zeros(3,1),[],'nonlinear_nonlcon');
disp('x的值为：')
disp(x);
disp('得到的最优值为：')
disp(y);