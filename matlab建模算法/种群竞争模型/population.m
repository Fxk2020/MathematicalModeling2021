%种群竞争模型
h=0.1;%所取时间点间隔
ts=[0:h:70];%时间间隔
x0=[10,10];%初始条件，种群的数量
opt=odeset('reltol',1e-6,'abstol',1e-9);%相对误差1e-6，绝对误差1e-9
[t,x]=ode45(@PopulationCompetition,ts,x0,opt);%使用5级4阶龙格—库塔公式计算

%两个种群的数量增长
plot(t,x(:,1),'r',t,x(:,2),'b','LineWidth',2),grid;
title('种群增长趋势');
xlabel('时间');
ylabel('种群数量');
legend('甲种群','乙种群') %可依次设置成你想要的名字

pause;%暂停作图，按任意键后继续做图

%两个种群的相对数量
plot(x(:,1),x(:,2),'b','LineWidth',2),grid%作相轨线
title('种群相对数量');
xlabel('乙种群数量');
ylabel('甲种群数量');