import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta5/kone5.csv', sep = ',', decimal='.')

df['hour'] = df['time'].astype(str).str.split(':').str.get(0)
print(df)

df2 = df.groupby(['date','hour'])['value'].mean().unstack()
print(df2.unstack())

df3 = df.groupby(['date','hour'])['value'].std().unstack()
print(df2.unstack())
#print(df3.unstack().iloc[:, 1])

#df4 = pd.merge(df2.unstack(), df3.unstack(), on = ['hour', 'date'], how = 'inner')
#df4 = df2.unstack.join(df3.unstack)
#print(df4)
df4 = pd.concat([df2, df3], axis=1)
#print(df4.iloc[:,20:30])
#print(df4.unstack())
#print(df4.stack())