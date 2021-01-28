clear;
%程序参数设定
%城市坐标Coordinates
Coord=...
[0.6683 0.6195 0.4 0.2439 0.1707 0.2293 0.5171 0.8732 0.6878 0.8488;
0.2536 0.2634 0.4439 0.1463 0.2293 0.761 0.9414 0.6536 0.5219 0.3609];
t0 = 1;%初温t0
iLk=20;%内循环的最大迭代系数
oLk=50;
lam=0.95;
istd=0.001;
ostd=0.001;
ilen=5;
olen=5;
%程序主体
m=length(Coord);
fare=distance(Coord);
path=1:m;
pathfar=pathfare(fare,path);
ores=zeros(1,olen);
e0=pathfar;
t=t0;
for out=1:oLk
ires=zeros(1,ilen);
for in=1:iLk
[newpath,v]=swap(path,1);
el=pathfare(fare,newpath);
%
r=min(1,exp(-(el-e0)/t));
if rand<r
path=newpath;
e0=el;
end
ires=[ires(2:end) e0];
%
if std(ires,1)<istd
break;
end
end
ores=[ores(2:end) e0];
if std(ores,1)<ostd
break;
end
t=lam*t;
end
pathfar = e0;
%
fprintf('近似最优路径为：\n');
disp(path)
fprintf('近似最优路径路程\tpathfare=');
disp(pathfar);
myplot(path,Coord,pathfar);