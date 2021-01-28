%定义非线性规划中的非线性约束条件
function [g,h] = nonlinear_nonlcon(x)
    g = [-x(1)^2+x(2)-x(3)^2
        x(1)+x(2)^2+x(3)^3-20];
    h = [-x(1)-x(2)^2+2
        x(2)+2*x(3)^2-3];
    