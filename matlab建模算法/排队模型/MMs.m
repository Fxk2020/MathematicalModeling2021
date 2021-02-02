function[Lq,L,W,Wq]=MMs(s,mu,lambda)

ro=lambda/mu;
ros=ro/s;
sum1=0;
for i=0:(s-1)
   sum1=sum1+ro.^i/factorial(i);
end

sum2=ro.^s/factorial(s)/(1-ros);

p0=1/(sum1+sum2);
p=ro.^s.*p0/factorial(s)/(1-ros);
Lq=p.*ros/(1-ros);
L=Lq+ro;
W=L/lambda;
Wq=Lq/lambda;
