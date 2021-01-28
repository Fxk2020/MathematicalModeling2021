%定义非线性规划中的目标函数
function f = nonlinear_f(x)
    f = sum(x.^2)+8;