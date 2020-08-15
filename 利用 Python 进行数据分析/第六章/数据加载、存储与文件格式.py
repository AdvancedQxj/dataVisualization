#!/usr/bin/env python
# coding: utf-8

# 6.1 读写文本格式的数据

# read_csv 和 read_table 函数

# In[1]:


import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[3]:


get_ipython().system('cat ex1.csv')


# In[4]:


df = pd.read_csv('examples/ex1.csv')
df


# In[6]:


pd.read_table('examples/ex1.csv', sep=',')


# 没有标题行的文件

# In[7]:


get_ipython().system('cat examples/ex2.csv')


# 列名可以默认,也可以自己定义

# In[8]:


pd.read_csv('examples/ex2.csv', header=None)


# In[9]:


pd.read_csv('examples/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])


# 将某一列作为 DataFrame 的索引

# In[10]:


names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('examples/ex2.csv', names=names, index_col='message')


# 将多个列做成一个层次化索引

# In[11]:


get_ipython().system('cat examples/csv_mindex.csv')


# In[12]:


parsed = pd.read_csv('examples/csv_mindex.csv',
                     index_col=['key1', 'key2'])
parsed


# 有一些表格可能不是用固定的分隔符去分割字段的

# In[13]:


list(open('examples/ex3.txt'))


# 传递一个正则表达式作为 read_table 的分隔符

# In[14]:


result = pd.read_table('examples/ex3.txt', sep='\s+')
result


# 可以使用 skiprows 跳过文件的第一行、第三行和第四行

# In[15]:


get_ipython().system('cat examples/ex4.csv')


# In[16]:


pd.read_csv('examples/ex4.csv', skiprows=[0, 2, 3])


# 缺失值标记

# In[17]:


get_ipython().system('cat examples/ex5.csv')


# In[18]:


result = pd.read_csv('examples/ex5.csv')
result


# In[19]:


pd.isnull(result)


# na_values 可以用一个列表或集合的字符串表示缺失值

# In[21]:


result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])
result


# 字典的各列可以使用不同的 NA 标记值

# In[22]:


sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('examples/ex5.csv', na_values=sentinels)


# 6.1.1 逐块读取文本文件

# 设置 pandas 显示的更紧一些

# In[23]:


pd.options.display.max_rows = 10


# In[25]:


result = pd.read_csv('examples/ex6.csv')
result


# 只显示指定行数

# In[26]:


pd.read_csv('examples/ex6.csv', nrows=5)


# 要逐块读取文件,可以指定 chunksize

# In[28]:


chunker = pd.read_csv('examples/ex6.csv', chunksize=1000)
chunker


# 迭代处理 ex6.csv,将值计数聚合到'key'列中

# In[29]:


tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)
tot[:10]


# 6.1.2 将数据写出到文本格式

# In[30]:


data = pd.read_csv('examples/ex5.csv')
data


# 利用 DataFrame 中的 to_csv 方法,将数据写到一个一都好分割的文件中

# In[31]:


data.to_csv('examples/out.csv')


# In[32]:


get_ipython().system('cat examples/out.csv')


# 也可以设置其他分隔符

# In[33]:


import sys
data.to_csv(sys.stdout, sep='|')


# 缺失值在输出结果中会被表示为空字符串

# In[34]:


data.to_csv(sys.stdout, na_rep='NULL')


# 禁用行和列的标签

# In[35]:


data.to_csv(sys.stdout, index=False, header=False)


# 只写出一部分的列,并按指定的顺序排列

# In[36]:


data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])


# Series 也有一个 to_csv 方法

# In[51]:


dates = pd.date_range(start='1/1/2000', periods=7)
ts = pd.Series(np.arange(7), index=dates)
ts.to_csv('examples/tseries.csv', header=False)
get_ipython().system('cat examples/tseries.csv')


# 6.1.3 处理分隔符格式

# In[52]:


get_ipython().system('cat examples/ex7.csv')


# 对于任何单字符分隔符文件,可以直接使用 Python 内置的 csv模块

# In[54]:


import csv
f = open('examples/ex7.csv')
reder = csv.reader(f)
for line in reder:
    print(line)


# 读取文件到一个多行的列表中

# In[55]:


with open('examples/ex7.csv') as f:
    lines = list(csv.reader(f))


# 将这些行分为标题行和数据行

# In[56]:


header, values = lines[0], lines[1:]


# 创建数据列字典

# In[57]:


data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict


# 定义 csv 文件的格式

# 6.1.4 JOSN 数据

# In[59]:


obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38,
               "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""


# In[60]:


import json
result = json.loads(obj)
result


# 将 Python 对象装换为 JOSN 格式

# In[61]:


asjson = json.dumps(result)


# 将 JSON 对象装换为 DataFrame

# In[62]:


siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
siblings


# pandas.read_json 可以自动将特别格式的 JSON 数据集装换为 Series 或 DataFrame

# In[63]:


get_ipython().system('cat examples/example.json')


# 默认假设 JSON 数组中每个对象是表格中的一行

# In[64]:


data = pd.read_json('examples/example.json')
data


# 从 pandas 输出到 JSON

# In[66]:


print(data.to_json())


# 6.1.5 XML 和 HTML:WEB 信息收集

# In[67]:


tables = pd.read_html('examples/fdic_failed_bank_list.html')
len(tables)


# In[68]:


failures = tables[0]
failures.head()


# 计算按年份计算倒闭的银行数

# In[69]:


close_timestamps = pd.to_datetime(failures['Closing Date'])
close_timestamps.dt.year.value_counts()


# 6.1.6 利用 lxml.objectify 解析 XML

# 6.2 二进制数据格式

# to_pickle 方法将数据以 pickle 格式保存到磁盘上.pickle用于短期存储,难以保证格式永远是稳定的

# In[70]:


frame = pd.read_csv('examples/ex1.csv')
frame


# In[71]:


frame.to_pickle('examples/frame_pickle')


# In[72]:


pd.read_pickle('examples/frame_pickle')


# 6.2.1 使用 HDF5 格式

# In[73]:


frame = pd.DataFrame({'a': np.random.randn(100)})
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store


# In[74]:


store['obj1']


# HDFStore 支持两张存储模式 fixed 和 table;后者较慢,但是支持使用特殊语法进行查询操作

# In[75]:


store.put('obj2', frame, format='table')
store.select('obj2', where=['index>=10 and index<=15'])


# In[76]:


store.close


# pandas.read_hdf 函数可以快捷使用以上工具

# In[83]:


frame.to_hdf('mydata.h5', 'obj3', format='table')
pd.read_hdf('mydata.h5', 'obj3', where=['index<5'])


# 6.2.2 读取 Microsoft Excel文件

# pandas.read_excel 函数支持读取存储 excel 中的表格型数据

# In[87]:


xlsx = pd.ExcelFile('examples/ex1.xlsx')
pd.read_excel(xlsx, 'Sheet1')


# In[88]:


frame = pd.read_excel('examples/ex1.xlsx', 'Sheet1')
frame


# 将 pandas 数据写入为 Excel 格式,首先创建一个 ExcelWriter,然后使用 pandas 对象的 to_excel 方法将数据写入其中

# In[89]:


writer = pd.ExcelWriter('examples/ex2.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()


# In[93]:


frame.to_excel('examples/ex3.xlsx')


# 6.3 Web APIs 交互

# 在 GitHub 搜索 30 个最新的 pandas 项目

# In[94]:


import requests
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
resp


# In[95]:


data = resp.json()
data[0]['title']


# In[96]:


issues = pd.DataFrame(data, columns=['number', 'title',
                                     'labels', 'state'])
issues


# 6.4 数据库交互

# In[97]:


import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""
con = sqlite3.connect('mydata.sqlite')
con.execute(query)


# In[98]:


con.commit()


# 插入数据

# In[99]:


data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()


# 查询数据

# In[100]:


cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows


# 数据导入 DataFrame

# In[102]:


cursor.description


# In[104]:


pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


# 使用 SQLAlchemy 连接 SQLite 数据库,并从之前创建的表读取数据

# In[106]:


import sqlalchemy as sqla
db = sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test', db)

