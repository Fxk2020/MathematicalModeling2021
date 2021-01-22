# Python画图-matplotlib

## 1.概述

### 1.1官网

[地址](https://matplotlib.org/)

[菜鸟教程](https://www.runoob.com/w3cnote/matplotlib-tutorial.html)

[中文官方文档](https://www.matplotlib.org.cn/)

### 1.2API

绘图方式分为：函数式和面向对象式

![image-20210117200953713](C:\Users\26292\AppData\Roaming\Typora\typora-user-images\image-20210117200953713.png)





##### pyplot

是一种函数式的绘图方式，对于更复杂的图，建议使用面向对象的API。

matplotlib.pyplot是matplotlib的基于状态的接口。它提供了类似于MATLAB的绘图方式。 pyplot主要用于交互式绘图和程序化绘图生成的简单情况：

例子绘制sinx的图像：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

![sin](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\概述\sin.png)

##### 函数



## Functions（了解即可）



### 2.图的性质

#### 2.1）图像的性质

```python
# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(x, y, color="r", linewidth=2.0, linestyle="-", label='图像名称')
```

color是图像的颜色；

```python
=============    ===============================
character        color
=============    ===============================
``'b'``          blue
``'g'``          green
``'r'``          red
``'c'``          cyan
``'m'``          magenta
``'y'``          yellow
``'k'``          black
``'w'``          white
=============    ===============================
```

linewidth是图像的宽度；

linestyle是图像的样式：

```python
``'-'``          实线
``'--'``         虚线
``'-.'``         点划线
``':'``          点线
```

label是图像名称需要配合legend（图的注释）使用

```python
legend(loc='upper left')
```

title是图的题目

```python
plt.title('各省GDP图')
```

设置xy轴的坐标名称

```python
plt.xlabel('省份')
plt.ylabel('GDP总量')
```

文本注释text（前两个参数是text坐标中x，y的值）

```python
plt.text(-1, 105000, '红色的为直辖市，蓝色的为省份')
```

在途中添加每个点的注释：

```python
for a, b in zip(year, GDP):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)  # fontsize表示柱坐标上显示字体的大小
```



#### 2.2）xy轴的性质

```python
# 设置xy轴的上下界
xmin, xmax = X.min(), X.max()
ymin, ymax = C.min(), C.max()

dx = (xmax - xmin) * 0.1
dy = (ymax - ymin) * 0.1

xlim(xmin - dx, xmax + dx)
ylim(ymin - dy, ymax + dy)

# 设置xy轴的记号点
xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])
```

改变坐标轴位置：

```python
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
```

没改之前：

![没改之前](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\图像性质\没改之前.png)

改了之后：



![改了之后](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\图像性质\改了之后.png)

设置xy轴的刻度

```python
x_major_locator = MultipleLocator(1)  # 把x,y轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(1)
axe.xaxis.set_major_locator(x_major_locator)  # 设置刻度
axe.yaxis.set_major_locator(y_major_locator)
```

设置x轴的偏斜度

```python
plt.xticks(year, year, rotation=60)  # 这里是调节横坐标的倾斜度，rotation是度数
```





#### 2.3)给图添加注释

```python
# plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
scatter([t, ], [np.cos(t), ], 50, color='blue')

annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
         xy=(t, np.sin(t)), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# plot([t, t], [0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
scatter([t, ], [np.sin(t), ], 50, color='red')

annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
```



scatter标记特殊的点；

annotate给点做注释。

![做注释](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\图像性质\做注释.png)

当我们调用 plot 函数的时候，matplotlib 调用 gca() 函数以及 gcf() 函数来获取当前的坐标轴和图像；如果无法获取图像，则会调用 figure() 函数来创建一个——严格地说，是用 subplot(1,1,1) 创建一个只有一个子图的图像。

### 3.绘图

#### 3.1）折线图

以中国总人口的变化为例进行绘图

##### 从Excel中读取数据

```java
import xlrd  # 用于读取Excel数据的包

def excel_data(file='population.xls'):
    """

    :param file:文件名
    :return: 年份和人口总数
    """
    try:
        # 打开Excel文件读取数据
        data = xlrd.open_workbook(file)
        # 获取第一个工作表
        table = data.sheet_by_index(0)
        # 获取行数
        nrows = table.nrows
        # 获取列数
        ncols = table.ncols
        # 定义excel_list
        date = []
        population = []
        for row in range(nrows):
            # 获取时间
            cell_value = table.cell(row, 0).value
            # 把时间追加到date中
            date.append(cell_value)
        for row in range(nrows):
            # 获取人口
            cell_value = table.cell(row, 1).value
            # 把人口追加到population中
            population.append(cell_value)
        return date, population
    except Exception as e:
        print(str(e))
```

进行绘图：

```python
date, population = excel_data()
# for i in population:
#     print(i)

fig, ax = plt.subplots()
ax.plot(date, population)

ax.set(xlabel='year (s)', ylabel='population (mV)',
       title='Demographic change map of the People‘s Republic of China')
ax.grid()
plt.show()
```



效果：

![出生率死亡率自然增长率](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\中国人口变化图\出生率死亡率自然增长率.png)![人口总数变化图](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\中国人口变化图\人口总数变化图.png)

#### 3.2）散点图scatter

[详细博客](https://blog.csdn.net/hustqb/article/details/77869076)

每年教育经费支出的python代码：

```python
def excel_data_education(file='education.xls'):
    # 获取第一个工作表 获取行数 获取列数
    table, nrows, ncols = excel_data(file)
    X = []
    Y = []
    for col in range(ncols):
        if col > 0:
            # 获取X
            cell_value = table.cell(2, col).value
            X.insert(0, cell_value)
    for col in range(ncols):
        if col > 0:
            # 获取Y
            cell_value = table.cell(3, col).value
            Y.insert(0, cell_value)
    return X, Y

def education():
    X, Y = excel_data_education()
    scatter(X, Y)
    show()
```



每年的教育经费支出：

![历年教育经费支出](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\散点图\历年教育经费支出.png)

散点图的性质：

```python
n = 32
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
area = np.random.normal(0, 1, n) * 1000  # 包含32个均匀分布的随机值的面积数组，大小[0, 1000]
scatter(X, Y, s=area, alpha=0.6)  # 绘制散点图，面积随机
```



![Figure_1](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\散点图\Figure_1.png)

#### 3.3）条形图（Bar graph）

csv文件：

逗号分隔值（Comma-Separated Values，CSV，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）。纯文本意味着该文件是一个字符序列，不含必须像二进制数字那样被解读的数据。CSV文件由任意数目的记录组成，记录间以某种换行符分隔；每条记录由字段组成，字段间的分隔符是其它字符或字符串，最常见的是逗号或制表符。通常，所有记录都有完全相同的字段序列.

显示y值得方法

```python
for a, b in zip(province, gdp_year):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
```

```python
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from numpy.polynomial.tests.test_laguerre import L2, L1

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


def excel_data(file):
    """
    :param file:文件名
    :return: 工作表，行数，列数
    """
    try:
        # 打开Excel文件读取数据
        data = xlrd.open_workbook(file)
        # 获取第一个工作表
        table = data.sheet_by_index(0)
        # 获取行数
        nrows = table.nrows
        # 获取列数
        ncols = table.ncols
        return table, nrows, ncols
    except Exception as e:
        print(str(e))


def excel_data_list(file):
    """
    :param file:文件名
    :return:
    """
    try:
        # 获取第一个工作表 获取行数 获取列数
        table, nrows, ncols = excel_data(file)
        year = []
        GDP = []
        location = []
        gdp_year = []
        for i in range(ncols):
            if 3 < i < 35:
                location.append(table.cell(i, 0).value)
                gdp_year.append(table.cell(i, 27).value)  # 更改列数可以获得不同年份的GDP 1-2019v2-2018
        for i in range(nrows):
            if 0 < i < 28:
                year.append(table.cell(3, i))
        return location, gdp_year, year
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    province, gdp_year, year = excel_data_list('Annualdatabyprovince.xls')
    color = [0]*len(province)
    for i in range(len(province)):
        if province[i] == '北京' or province[i] == '上海' or province[i] == '重庆' or province[i] == '天津':
            color[i] = 'r'
        else:
            color[i] = 'b'

    plt.bar(province, gdp_year, color=color, width=0.7)
    plt.title('1993年各省GDP图')
    plt.xlabel('省份')
    plt.ylabel('GDP总量')
    plt.text(-1, 105000, '红色的为直辖市，蓝色的为省份')
    for a, b in zip(province, gdp_year):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    plt.show()

```

#### 3.4）等高线图（contourf/contour）

x,y是坐标

z是高度

contourf是填充颜色

contour是等高线

```
# 导入模块
import numpy as np
import matplotlib.pyplot as plt

# 建立步长为0.01，即每隔0.01取一个点
step = 0.01
x = np.arange(-10, 10, step)
y = np.arange(-10, 10, step)
# 也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份

# 将原始数据变成网格数据形式
X, Y = np.meshgrid(x, y)

# 写入函数，z是大写
Z = X ** 2 + Y ** 2
# 设置打开画布大小,长10，宽6
# plt.figure(figsize=(10,6))
# 填充颜色，f即filled
plt.contourf(X, Y, Z, 8, alpha=0.8, cmap='jet')
# 画等高线
plt.contour(X, Y, Z, 8, colors='black')
plt.show()

```

#### 3.5）饼状图（pie）

```python
plt.figure(figsize=(6,9)) #调节图形大小
labels = [u'大型',u'中型',u'小型',u'微型'] #定义标签
sizes = [46,253,321,66] #每块值
colors = ['red','yellowgreen','lightskyblue','yellow'] #每块颜色定义
explode = (0,0,0,0) #将某一块分割出来，值越大分割出的间隙越大
patches,text1,text2 = plt.pie(sizes,
                      explode=explode,
                      labels=labels,
                      colors=colors,
                      autopct = '%3.2f%%', #数值保留固定小数位
                      shadow = False, #无阴影设置
                      startangle =90, #逆时针起始角度设置
                      pctdistance = 0.6) #数值距圆心半径倍数距离
#patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部的文本
# x，y轴刻度设置一致，保证饼图为圆形
plt.axis('equal')
plt.show()

```



python代码，历年男女比例图

```python
import matplotlib.pyplot as plt
import xlrd

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def excel_data(file):
    """
    :param file:文件名
    :return: 工作表，行数，列数
    """
    try:
        # 打开Excel文件读取数据
        data = xlrd.open_workbook(file)
        # 获取第一个工作表
        table = data.sheet_by_index(0)
        # 获取行数
        nrows = table.nrows
        # 获取列数
        ncols = table.ncols
        return table, nrows, ncols
    except Exception as e:
        print(str(e))


def excel_data_list(file, year_is):
    """
    :param file:文件名
    :return:
    """
    try:
        # 获取第一个工作表 获取行数 获取列数
        table, nrows, ncols = excel_data(file)
        year = 0
        girl_number = 0
        boy_number = 0
        if 1949 <= year_is <= 2019:
            year_is = 2020 - year_is
            year = table.cell(2, year_is).value
            boy_number = table.cell(4, year_is).value
            girl_number = table.cell(5, year_is).value
        else:
            print("输入的年份有误")
        return year, girl_number, boy_number
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    for i in range(71):
        year_what = 1949 + i
        year1, girl_number1, boy_number1 = excel_data_list('yearData.xls', year_what)
        labels = ['男性人口', '女性人口']
        sizes = [boy_number1, girl_number1]
        colors = ['b', 'r']
        explode = (0, 0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%.3f%%',  # 数值保留固定小数位
                shadow=False,  # 无阴影设置
                startangle=90,  # 逆时针起始角度设置
                pctdistance=0.6)  # 数值距圆心半径倍数距离)
        plt.axis('equal')
        plt.title(str(year_what) + '年男女比例图')
        plt.show()
```



![1949](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\饼状图\1949.png)

#### 3.6）3D 图！！！（Axes3D）

基本做$z=x^2+y^2$的3D图：

```python
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

show()
```



![z2=x2+y2](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\3D图\z2=x2+y2.png)

##### （0）创建三维画布

```python
#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图


#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax2 = Axes3D(fig)
```



##### （1）三维散点图图

要点：一个xy对应一个z值，也就是x，y和z的数据总量是相同的。

```python
ax = Axes3D(fig)
ax.set_xlabel('时间')
ax.set_ylabel('国家（r-中国，b-印度，g-美国，w-俄罗斯，black-日本，orange-德国）')
ax.set_zlabel('人口总数（千万）')
ax.text2D(0.1, 0.9, "世界人口散点图（1960-2019年）", transform=ax.transAxes)
ax.scatter3D(year_is, y, population, color=color)  # 绘制散点图
```

##### （2）三维曲线图

要点：还是一个xy对应一个z值，也就是x，y和z的数据总量是相同的。

```python
fig = plt.figure()
axe = Axes3D(fig)
axe.set_xlabel('年份')
axe.set_ylabel('季度')
axe.set_zlabel('GDP总量')
axe.text2D(0.1, 0.9, "中国GDP总量季度图（2002-2020年）", transform=axe.transAxes)

x_major_locator = MultipleLocator(1)  # 把x,y轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(1)
axe.xaxis.set_major_locator(x_major_locator)  # 设置刻度
axe.yaxis.set_major_locator(y_major_locator)

axe.plot3D(year, quarterly, GDP, 'r')
plt.show()
```

![GDP](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\3D图\曲线图\GDP.png)

但是这个3维图不如2维图直观：

![二维图](C:\Users\26292\Desktop\美赛冲冲冲\ModelingStudy\Python画图\3D图\曲线图\二维图.png)

可以明显看出2002年到2020年中国经济的增长，同时可以看到每年的第四季度总是最高的，之后来年的第一季度会下降很多。并且近几年收到新冠疫情和中美贸易战的影响，GDP浮动较大。

##### （3）三维曲面

这个图是比较高级的图，一般在高等数学的学习和人工智能领域会有所使用，比如做出某个复杂函数的图形。

根据三点数据处理成xy网格数据

```python
# 定义三维数据
xx = np.arange(-5, 5, 0.5)
yy = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(xx, yy)
```

网格的处理过程：

```python 
# xx 和 yy 均为:
array([-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5])

# meshgrid 函数后
## X,Y
[[-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]
 [-5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5
   2.   2.5  3.   3.5  4.   4.5]]

```

画出3维图像：

```python
axe.plot_surface(X, Y, Z, rstride=1,  # row 行步长
                 cstride=2,  # colum 列步长
                 cmap='rainbow')
```

添加平面的投影：

```python
axe.contourf(X, Y, Z,
             zdir='z',  # 使用数据方向
             offset=-2,  # 填充投影轮廓位置
             cmap='rainbow')
```

#### 3.7）热度图

imshow：热图（heatmap）是数据分析的常用方法，通过色差、亮度来展示数据的差异、易于理解。Python在Matplotlib库中，调用imshow()函数实现热图绘制。

```python
plt.imshow(
    X,
    cmap=None,
    norm=None,
    aspect=None,
    interpolation=None,
    alpha=None,
    vmin=None,
    vmax=None,
    origin=None,
    extent=None,
    shape=None,
    filternorm=1,
    filterrad=4.0,
    imlim=None,
    resample=None,
    url=None,
    *,
    data=None,
    **kwargs,
)
```

参数X：图像数据，支持数组形状

参数cmap:将标量数据映射到色彩图

参数norm：Normalize会对其进行缩放[0,1]的数据值内

参数aspect：控制轴的纵横比。该参数可能使图像失真，即像素不是方形的

参数interpolation：插值方法

参数alpha：透明度

参数origin：数组的[0,0]索引放在轴的左上角或左下角。{'upper', 'lower'}

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
print(x_train.shape)
plt.figure()
plt.imshow(x_train[4], cmap='hot', alpha=0.7, origin='lower')
plt.colorbar()
plt.grid(False)
plt.show()

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test,  y_test, verbose=2)

```

