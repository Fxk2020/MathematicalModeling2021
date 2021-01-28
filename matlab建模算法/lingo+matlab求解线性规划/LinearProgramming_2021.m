%求解线性规划的程序
clc,clear
a=0;
hold on
while a<0.05
    f=[-0.05,-0.27,-0.19,-0.185,-0.185];

    A=[zeros(4,1),diag([0.025, 0.015, 0.055, 0.026])];
    b=a*ones(4,1);

    Aeq=[1,1.01,1.02,1.045,1.065];
    beq=1;

    Lb=zeros(5,1);

    [x,fval]=linprog(f,A,b,Aeq,beq,Lb);
    Q = -fval;%获取最大收益
    plot(a,Q,'*k');
    a=a+0.001;
end
xlabel('a(最大能承受的风险)'),ylabel('Q（收益）');

