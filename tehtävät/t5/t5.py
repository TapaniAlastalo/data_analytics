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

df5 = df.groupby(['date','hour'])['value'].std() / df.groupby(['date','hour'])['value'].mean()
df5.fillna(0, inplace=True)
print(df5.unstack())

print(df5.count())

#df5.plot.density()  # tässä luokat määritellään automaattisesti

#sns.pairplot(data = df, vars = ['suunta', 'tunti', 'nopeus'], hue = 'ajoneuvoluokka', diag_kind = 'kde', height = 4)
sns.distplot(df5, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = "reino")
plt.show()