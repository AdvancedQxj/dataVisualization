#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import Series, DataFrame


# 5.1 pandas 的数据结构介绍

# 5.1.1 Series

# Series 是一种类似于一维数组的对象,它由一组数据以及一组与之相关的数据标签(索引)组成

# In[30]:


obj = pd.Series([4, 7, -5, 3])
obj


# 通过 Series 的 values 和 index 属性获取数组表示形式和索引对象

# In[4]:


obj.values


# In[5]:


obj.index


# 可以设置索引

# In[6]:


obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# In[7]:


obj2.index


# 通过索引选取 Series 中的值

# In[8]:


obj2['a']


# In[9]:


obj2['d'] = 6


# In[11]:


obj2[['c', 'a', 'd']]


# 可以使用与 numpy 类似的运算

# In[12]:


obj2[obj2>0]


# In[13]:


obj2 * 2


# In[15]:


import numpy as np
np.exp(obj2)


# 可以将 Series 看做一个定长的有序字典,可以用在许多原本需要字典参数的函数中

# In[16]:


'b' in obj2


# In[17]:


'e' in obj2


# 可以使用字典创建 Series

# In[19]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3


# 可以设置索引

# In[20]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4


# pandas 的 isnull 和 notnull 函数可用于检测缺失数据

# In[21]:


pd.isnull(obj4)


# In[22]:


pd.notnull(obj4)


# Series 也有类似的方法

# In[23]:


obj4.isnull()


# Series 会根据运算的索引标签自动对齐数据

# In[24]:


obj3


# In[25]:


obj4


# In[26]:


obj3 + obj4


# Series 对象及其索引都有 name 属性

# In[27]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# 索引可以通过赋值的方式修改

# In[28]:


obj


# In[29]:


obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj


# 5.1.2 DataFrame

# DataFrame 是一个表格型的数据结构

# 通常通过传入一个由等长列表或 numpy 数组组成的字典

# In[32]:


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
frame


# 显示前 2 行数据

# In[33]:


frame.head()


# 指定 DataFrame 列的顺序

# In[34]:


pd.DataFrame(data, columns=['year', 'state', 'pop'])


# 如果传入的列在数据中找不到,就会产生缺失值

# In[36]:


frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'], 
                      index=['one', 'two', 'three', 'four', 'five', 'six'])
frame2


# 将 DateFrame 的列获取为一个 Series

# In[37]:


frame2['state']


# 使用 loc获取 DateFrame 的行

# In[38]:


frame2.loc['three']


# 修改列的值

# In[40]:


frame2['debt'] = 16.5
frame2


# In[42]:


frame2['debt'] = np.arange(6.)
frame2


# 赋值的是一个 Series 会精确匹配索引,所有的空位都将填上缺失值

# In[44]:


val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2


# del 删除列

# In[45]:


frame2['test'] = frame2.state == 'Ohio'
frame2


# In[46]:


del frame2['test']
frame2


# 嵌套字典传给 DateFrame, pandas 会将外层字典的键作为列,内层键则作为行索引

# In[47]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
frame3


# 转置

# In[48]:


frame3.T


# 设置 index 和 columns 的 name 属性

# In[50]:


frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3


# 只获取其中的数据

# In[51]:


frame3.values


# 5.1.3 索引对象

# 构建 Series 和 DataFrame 时,所用到的标签都会变为一个 index

# In[53]:


obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index


# index对象是不可变的,不能修改;可以使index 对象在多个数据结构之间安全共享

# In[55]:


labels = pd.Index(np.arange(3))
labels


# In[56]:


obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2


# In[58]:


obj2.index is labels


# 5.2 基本功能

# 5.2.1 重新索引

# pandas 对象的一个重要方法 reindex,它创建一个新对象,它的数据符合新的索引
# reindex 将会根据新索引进行重排,某个索引值当前不存在,就会引入缺失值

# In[59]:


obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj


# In[60]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# 对于时间序列这样的有序数据,重新索引时可能需要做一些插值处理.method 选项可以达到此目的,例如使用 ffill 可以实现前向值填充

# In[61]:


obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3


# In[62]:


obj3.reindex(range(6), method='ffill')


# reindex 可以修改行(列)索引

# In[64]:


frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame


# In[65]:


frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2


# 列可以用 columns 关键字来修改

# In[67]:


states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)


# 5.2.2 丢弃指定轴上的项

# drop 方法返回的是一个在指定轴上删除了指定值的新对象

# In[68]:


obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj


# In[70]:


new_obj = obj.drop('c')
new_obj


# In[71]:


obj.drop(['d', 'c'])


# 对于 DateFrame 可以删除任意轴上的索引值

# In[72]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# 删除行

# In[73]:


data.drop(['Colorado', 'Ohio'])


# 删除列

# In[74]:


data.drop('two', axis=1)#默认 axis=0 表示行


# In[75]:


data.drop(['two', 'four'], axis=1)


# inplace 会销毁所有被删除的数据

# In[76]:


obj.drop('c', inplace=True)
obj


# 5.2.3 索引、选取和过滤

# 索引

# In[77]:


obj = pd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
obj


# In[78]:


obj['b']


# In[79]:


obj[1]


# In[80]:


obj[2:4]


# In[82]:


obj[['b', 'a', 'd']]


# In[83]:


obj[[1, 3]]


# In[84]:


obj[obj<2]


# 利用标签切片,包含末端

# In[86]:


obj['b':'c'] = 5
obj


# 获取一个或多个列

# In[87]:


data = pd.DataFrame(np.arange(16).reshape(4, 4),
                   index=['Ohio', 'Colorado', 'Utah', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
data


# In[88]:


data['two']


# In[89]:


data[['three', 'one']]


# 特殊情况

# In[90]:


data[:2]


# In[91]:


data[data['three']>5]


# In[92]:


data < 5


# In[94]:


data[data<5] = 0
data


# 5.2.4 用 loc 和 iloc 进行选取

# 轴标签loc,整数索引 iloc

# In[95]:


data.loc['Colorado', ['two', 'three']]


# In[97]:


data.iloc[2, [3, 0, 1]]


# In[98]:


data.iloc[2]


# In[99]:


data.iloc[[1, 2], [3, 0, 1]]


# 这两个索引函数也适用于一个标签或多个标签的切片

# In[101]:


data.loc[:'Utah', 'two']


# In[105]:


data.iloc[:, :3][data.three>5]


# In[103]:


data


# 5.2.5 整数索引

# In[107]:


ser = pd.Series(np.arange(3.))
ser


# ser[-1]会出错,索引只包含 0,1,2
# 将索引设置为 (a,b,c),ser[-1]会成功

# In[109]:


ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
ser2


# In[110]:


ser2[-1]


# 索引为了更精确请适用 loc 或 iloc

# In[111]:


ser[:1]


# In[112]:


ser.loc[:1]


# In[113]:


ser.iloc[:1]


# 5.2.6 算术运算和数据对齐

# 存在不同的索引对,则结果的索引就是该索引对的并集

# In[114]:


s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=['a', 'c', 'e', 'f', 'g'])
s1


# In[115]:


s2


# In[116]:


s1 + s2


# 对于 DataFrame,对齐操作会同时发生在行和列上

# In[117]:


df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1


# In[118]:


df2


# In[119]:


df1 + df2


# In[120]:


df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})
df1


# In[121]:


df2


# In[122]:


df1 - df2


# 5.2.7 在算术方法中填充值

# In[124]:


df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                   columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                   columns=list('abcde'))
df2.loc[1, 'b'] = np.nan
df1


# In[125]:


df2


# In[126]:


df1 + df2


# In[127]:


df1.add(df2, fill_value=0)#fill_value填充值


# In[128]:


1 / df1


# In[129]:


df1.rdiv(1)


# In[130]:


df1.reindex(columns=df2.columns, fill_value=0)


# 5.2.8 DataFrame 和 Series 之间的运算

# 计算一个二维数组与某某行之间的差,二维数组的每一行都去减这一行

# In[131]:


arr = np.arange(12).reshape(3, 4)
arr


# In[132]:


arr[0]


# In[133]:


arr - arr[0]


# In[134]:


frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame


# In[135]:


series


# In[136]:


frame - series


# In[137]:


series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame + series2


# 匹配行且在列上广播,则必须使用算术运算方法

# In[138]:


series3 = frame['d']
frame


# In[139]:


series3


# In[144]:


frame.sub(series3, axis=0)


# 5.2.9 函数应用和映射

# 元素级数组方法 ufuncs

# In[145]:


frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# 将函数应用到由各列或行形成的一维数组上,使用 apply 方法实现

# In[146]:


f = lambda x: x.max() - x.min()
frame.apply(f)


# In[151]:


frame.apply(f, axis=1)


# apply 的函数可以返回多个值组成的 Series

# In[152]:


def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)


# 获得 frame 中各个浮点值的格式化字符串,使用 applymap

# In[153]:


format = lambda x: '%.2f' % x
frame.applymap(format)


# 应用元素级函数 map

# In[154]:


frame['e'].map(format)


# 5.2.10 排序和排名

# 对索引排序

# In[155]:


obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()


# 对于 DataFrame 可以对任意一个轴上的索引进行排序

# In[156]:


frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame.sort_index()


# In[159]:


frame.sort_index(1)


# 数据排序默认升序,也可以降序

# In[160]:


frame.sort_index(ascending=False, axis=1)


# 若按值对 Series 进行排序,也可以使用 sort_values方法

# In[161]:


obj = pd.Series([4, 7, -3, 2])
obj.sort_values()


# 排序时,缺失值或被放到 Series 的末尾

# In[162]:


obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()


# 排序 DataFrame 时,可以根据一个或多个列中的值进行排序

# In[163]:


frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# In[164]:


frame.sort_values(by='b')


# In[165]:


frame.sort_values(by=['a', 'b'])


# 平均排名 rank

# In[166]:


obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()


# 根据值在原数据中出现的顺序给出排名

# In[167]:


obj.rank(method='first')


# 降序排名且排名向前并列

# In[168]:


obj.rank(method='max',ascending=False)


# DataFrame 计算排名

# In[176]:


frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 
                      'a': [0, 1, 0, 1],
                      'c': [-2, 5, 8, -2.5]})
frame


# In[177]:


frame.rank(axis='columns')


# In[175]:


frame.rank(0)


# 5.2.11 带有重复标签的轴索引

# In[178]:


obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj


# is_unique 检查值是否唯一

# In[179]:


obj.index.is_unique


# 索引对应几个值就返回几个值

# In[180]:


obj['a']


# In[181]:


obj['c']


# DataFrame 一样如此

# In[182]:


df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df


# In[183]:


df.loc['a']


# 5.3 汇总和计算描述统计

# In[184]:


df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df


# DataFrame 的 sum 方法返回一个包含各列和的 Series

# In[185]:


df.sum()


# 按行求和

# In[186]:


df.sum(1)


# NaN 会自动排除,可以使用 skipna 选项禁用该功能

# In[187]:


df.mean(axis=1, skipna=False)


# 最大(小)值的索引

# In[189]:


df.idxmax()


# 累计型方法

# In[190]:


df.cumsum()


# describe 一次产生多个汇总统计

# In[191]:


df.describe()


# 对于非数值型数据,describe 会产生另外一种汇总统计

# In[192]:


obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()


# 5.4 相关系数与协方差

# 5.5 唯一值、值计数以及成员资格

# In[193]:


obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj


# 得到 Series 中的唯一值数组

# In[196]:


uniques = obj.unique()
uniques


# 计算 Series 中各值出现的频率

# In[197]:


obj.value_counts()


# isin 可以过滤 Series 中或 DataFrame 列中数据的子集

# In[198]:


obj


# In[200]:


mask = obj.isin(['b', 'c'])
mask


# In[201]:


obj[mask]


# 返回一个索引数组,从可能包含重复值的数组到领一个不同值的数组

# In[202]:


to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = pd.Series(['c', 'b', 'a'])
pd.Index(unique_vals).get_indexer(to_match)

