#!/usr/bin/env python
# coding: utf-8

# 4.1 NumPy 的 ndarray: 一种多维数组对象

# 4.1.1引入 Numpy,然后生成一个包含随机数据的小数组

# In[3]:


import numpy as np


# In[4]:


data = np.random.randn(2, 3)
data


# 进行数学运算

# In[5]:


data * 10


# In[6]:


data + data


# 表示各维度大小的元组

# In[7]:


data.shape


# 说明数组数据类型的对象

# In[8]:


data.dtype


# 使用 array 函数创建 ndarray

# In[9]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1


# 嵌套序列将会被转换为一个多维数组

# In[10]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# 检查数组 arr2 的维度是多少以及各维度大小

# In[12]:


arr2.ndim


# In[13]:


arr2.shape


# In[14]:


arr1.dtype


# In[15]:


arr2.dtype


# 创建全为 0 的数组

# In[16]:


np.zeros(10)


# In[17]:


np.zeros((3, 6))


# In[18]:


np.zeros((2, 3, 2))


# arange 是 Python 内置函数 range 的数组版

# In[19]:


np.arange(15)


# 4.1.2ndarray 的数据类型

# 创建数组时设置数据类型

# In[21]:


arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype


# In[22]:


arr2.dtype


# 通过 ndarray 的 astype 方法明确地将一个数组从一个 dtype 转换成另一个 dtype

# In[24]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype


# In[25]:


float_arr = arr.astype(np.float64)
float_arr.dtype


# 浮点数转换为整数,小数部分会被删除

# In[26]:


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.5])
arr


# In[27]:


arr.astype(np.int32)


# 某字符串数组表示的全是数字,也可以用 astype 转换为数组形式

# In[29]:


numeric_strings = np.array(['1.25', '-9.6'])
numeric_strings.astype(np.float64)


# 利用存在的数组类型进行数组类型转换

# In[30]:


int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380])
int_array.astype(calibers.dtype)


# 4.1.3 NumPy 数组的运算

# 大小相等的数组之间的任何算术运算都会将运算应用到元素级

# In[31]:


arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr


# In[32]:


arr * arr


# In[33]:


arr - arr


# 数组与标量的算术运算会将标量值传播到各个元素

# In[34]:


1 / arr


# In[35]:


arr ** 0.5


# 大小相同的数组之间的比较会生成布尔值数组

# In[36]:


arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2


# In[37]:


arr2 > arr


# 4.1.4 基本的索引和切片

# 表面上看,它们跟 Python 列表的功能差不多

# In[38]:


arr = np.arange(10)
arr


# In[39]:


arr[5]


# In[40]:


arr[5:8]


# In[42]:


arr[5:8] = 12
arr


# 上一例子数组切片是原始数组的视图,视图上的任何修改都会直接放映到源数组上

# 创建一个 arr 的切片

# In[44]:


arr_slice = arr[5:8]
arr_slice


# 当修改 arr_slice 中的值,原始数组 arr 中的值也会改变

# In[45]:


arr_slice[1] = 12345
arr


# 在二维数组中,各索引位置上的元素不再是标量而是一维数组

# In[49]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]


# 访问单个元素的两种方式

# In[50]:


arr2d[0][2]


# In[51]:


arr2d[0, 2]


# 在多维数组中,省略后面的索引,则返回对象会是一个维度低一些的数组

# In[52]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
arr3d


# In[53]:


arr3d[0]


# 标量值和数组都可以赋值给 arr3d[0]

# In[57]:


old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d


# In[58]:


arr3d[0] = old_values
arr3d


# 4.1.5切片索引

# ndarray 的切片语法跟 Python 列表这样的一维对象差不多

# In[59]:


arr


# In[60]:


arr[1:6]


# 二维数组稍显不同

# In[61]:


arr2d


# In[62]:


arr2d[:2]


# 传入多个切片

# In[63]:


arr2d[:2, 1:]


# 对切片表达式的赋值操作也会扩散到整个选区

# In[64]:


arr2d[:2, 1:] = 0
arr2d


# 4.1.6 布尔型索引

# In[65]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7,4)
names


# In[67]:


data


# 对 names 和字符串'Bob'的比较运算会产生一个布尔型数组

# In[68]:


names == 'Bob'


# 这个布尔型数组可用于数组索引,两个数组的长度需一致

# In[69]:


data[names == 'Bob']


# 再选取列

# In[70]:


data[names == 'Bob', 2:]


# In[72]:


data[names == 'Bob', 3]


# 其他与上例类似操作

# In[73]:


names != 'Bob'


# In[74]:


data[~(names == 'Bob')]


# In[75]:


cond = names == 'Bob'
data[~cond]


# In[76]:


mask = (names == 'Bob') | (names == 'Will')
mask


# In[77]:


data[mask]


# 将所有小于 0 的数据置 0

# In[79]:


data[data < 0] = 0
data


# 通过布尔数组设置整行整列的值

# In[81]:


data[names != 'Joe'] = 7
data


# 4.1.7 花式索引

# 利用整数数组进行索引,它会将数据复制到新数组中

# In[85]:


arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr


# 为了以特定顺序选取行子集,只需传入一个用于指定顺序的整数列表或 ndarray

# In[86]:


arr[[4, 3, 0, 6]]


# 使用负数索引将会从末尾开始选取行

# In[87]:


arr[[-3, -5, -7]]


# 一次传入多个索引数组会返回一个一维数组,选取的元素(1,0),(5,3),(7,1)和(2,2)

# In[88]:


arr = np.arange(32).reshape((8, 4))
arr


# In[89]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# 返回一个矩形区域

# In[90]:


arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# 4.1.8 数组转置和轴对换

# 转置是重塑的一种特殊形式,它返回的是原始视图,不会进行复制操作

# In[92]:


arr = np.arange(15).reshape((3, 5))
arr


# In[93]:


arr.T


# 利用 np.dot 计算矩阵内积

# In[95]:


arr = np.random.randn(6,3)
arr


# In[97]:


np.dot(arr.T, arr)


# 对于高维数组,transpose 需要得到一个由轴编号组成的元组才能对这些轴进行转置(元素不动,轴交换位置)

# In[98]:


arr = np.arange(16).reshape((2, 2, 4))
arr


# In[99]:


arr.transpose((1, 0 ,2))


# 只进行两个轴交换,可以使用 swapaxes 方法,需要接受一对轴编号

# In[100]:


arr.swapaxes(1,2)


# 4.2 通用函数:快递的元素级数组函数

# 通用函数时一种对 ndarray 中的数据执行元素级运算的函数,可以将其看做简单函数的矢量化包装器,如 sqrt 和 exp

# In[101]:


arr = np.arange(10)
arr


# In[102]:


np.sqrt(arr)


# In[103]:


np.exp(arr)


# 上面的例子都是一个数组的,还有一些接受两个数组的

# In[104]:


x = np.random.randn(8)
y = np.random.randn(8)


# In[105]:


x


# In[106]:


y


# In[107]:


np.minimum(x, y)


# 还有一些可以返回多个数组.modf是 Python 内置函数 divmod 的矢量化版本,它会返回浮点数数组的小数和整数部分

# In[108]:


arr = np.random.randn(7) * 5
arr


# In[109]:


remainder, whole_part = np.modf(arr)
remainder


# In[110]:


whole_part


# 4.3 利用数组进行数据处理

# np.meshgrid 函数接受两个一维数组,并产生两个二维矩阵

# In[111]:


points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
ys


# 进行简单计算

# In[112]:


z = np.sqrt(xs ** 2 + ys ** 2)
z


# 利用 matplotlib 创建这个二维数组的可视化视图

# In[115]:


import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# 4.3.1 将条件逻辑表述为数组运算

# numpy.where 函数时三元表达式 x if condition else y 的矢量化版本

# In[117]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result


# 上例缺点:处理速度不是很快,由纯 Python完成;无法用于多维数组
# 使用 np.where 更简洁

# In[119]:


result = np.where(cond, xarr, yarr)
result


# np.where 的后两个参数可以使标量

# In[120]:


arr = np.random.randn(4,4)
arr


# In[121]:


arr > 0


# In[122]:


np.where(arr>0, 2, -2)


# 标量和数组结合起来

# In[123]:


np.where(arr>0, 2, arr)


# 4.3.2 数学统计方法

# 聚类统计

# In[124]:


arr = np.random.randn(5, 4)
arr


# In[125]:


arr.mean()


# In[126]:


np.mean(arr)


# In[128]:


arr.sum()


# 统计设置 axis 参数的值,计算该轴向上的统计值
# 0:列
# 1:行

# In[129]:


arr.mean(axis=1)


# In[130]:


arr.sum(axis=0)


# cumsum累加函数,cumprod 累积函数

# In[131]:


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr


# In[132]:


arr.cumsum(axis=0)


# In[133]:


arr.cumprod(axis=1)


# 4.3.3 用于布尔型数组的方法

# 布尔值会被强制转换为 1 和 0
# sum 经常被用来对布尔型数组中的 true 值计算

# In[135]:


arr = np.random.randn(100)
(arr>0).sum()


# any 测试数组中是否存在一个或多个 True,而 all 则检测数组中所有值是否都是 True

# In[136]:


(arr>0).any()


# In[137]:


(arr>0).all()


# 4.3.4 排序

# In[138]:


arr = np.random.randn(6)
arr


# In[140]:


arr.sort()
arr


# 多维数组可以在任何一个轴向上进行排序,只需将轴编号传给 sort

# In[142]:


arr = np.random.randn(5, 3)
arr


# In[144]:


arr.sort(1)
arr


# np.sort 返回的数组是已排序的数组,会修改数组本身
# 计算数组分位数最简单的办法是对其进行排序,然后选取特定位置的值

# In[145]:


large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]


# 4.3.5 唯一化以及其他的集合逻辑

# np.unique 找出数组中的唯一值并返回已排序的结果

# In[146]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[147]:


ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# np.in1d 用于测试一个数组中的值在另外一个数组中的成员资格,返回一个布尔型数组

# In[148]:


values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# 4.4 用于数组的文件输入输出

# np.save 和 np.load 是读写磁盘数组数据的两个主要函数.默认情况下,数组是以未压缩的原始二进制格式保存在扩展名为 .npy 的文件中

# In[149]:


arr = np.arange(10)
np.save('some_arr', arr)


# 没有扩展名,会自动加上,然后通过 np.load 读取磁盘上的数组

# In[150]:


np.load('some_arr.npy')


# np.savez 可以将多个数组保存到一个未压缩文件中,将数组以关键字参数的形式传入即可

# In[155]:


np.savez('array_archive.npz', a=arr, b=arr)


# 加载 .npz 文件时,会得到一个类似字典的对象,该对象会对各个数组进行延迟加载

# In[156]:


arch = np.load('array_archive.npz')
arch['b']


# 如果数据压缩得很好,可以使用 numpy.savez_compressed

# In[157]:


np.savez_compressed('array_archive.npz')


# 4.5 线性代数

# 计算矩阵内积函数 dot

# In[158]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x


# In[159]:


y


# In[160]:


x.dot(y)


# x.dot(y) 等价于 np.dot(x, y)

# In[161]:


np.dot(x, y)


# numpy.linalg 中有一组标准的矩阵分解运算以及诸如求逆和行列式之类的东西

# In[162]:


from numpy.linalg import inv, qr
x = np.random.randn(5, 5)
mat = x.T.dot(x)
inv(mat)


# In[163]:


mat.dot(inv(mat))


# In[165]:


q, r = qr(mat)
r


# In[166]:


q


# 4.6 伪随机生成

# numpy.random.normal 标准正态分布的数组

# In[167]:


samples = np.random.normal(size=(4, 4))
samples


# 使用 numpy 的 np.random.seed 设置随机数生成种子

# In[168]:


np.random.seed(1234)


# numpy.random 的数据生成函数使用了全局的随机种子,要避免全局状态,可以使用 numpy.random.RandomState 创建一个与其他隔离的随机数生成器

# In[169]:


rng = np.random.RandomState(1234)
rng.randn(10)


# 4.7 示例:随机漫步

# 以纯 Pythond 的方式实现

# In[171]:


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
plt.plot(walk[:100])


# 以 np.random 模块实现

# In[172]:


nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
setps = np.where(draws>0, 1, -1)
walk = setps.cumsum()
plt.plot(walk[:100])


# 简单统计工作:最大值和最小值

# In[173]:


walk.max()


# In[174]:


walk.min()


# 使用函数 argmax 求第一次到达 10 或者 -10 的索引
# 注:argmax 会扫描整个数组,效率不高

# In[192]:


(np.abs(walk)>=10).argmax()


# 4.7.1 一次模拟多个随机漫步

# In[182]:


nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws>0, 1, -1)
walks = steps.cumsum(1)
walks


# 求最大值和最小值

# In[183]:


walks.max()


# In[184]:


walks.min()


# 检查哪些点到达 30 或者 -30

# In[185]:


hits30 = (np.abs(walks)>=30).any(1)
hits30


# In[186]:


hits30.sum()


# 获取达到 30 或者 -30 的点

# In[189]:


crossing_times = (np.abs(walks[hits30])>=30).argmax(1)
crossing_times.mean()


# In[190]:


crossing_times


# 使用 normal 生成指定均值和标准差的正态分布数据

# In[193]:


steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
steps

