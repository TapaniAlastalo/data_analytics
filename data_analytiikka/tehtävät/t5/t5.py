import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta5/kone5.csv', sep = ',', decimal='.')

df['hour'] = df['time'].astype(str).str.split(':').str.get(0)
#print(df)

df2 = df.groupby(['date','hour'])['value'].std() / df.groupby(['date','hour'])['value'].mean()
df2.dropna(inplace=True)
df2 = df2.reset_index()
df2.rename({'value': 'COV'}, axis=1, inplace=True)

df2['rank'] = df2['COV'].rank()
mMax = df2['rank'].max()
mMin = df2['rank'].min()

df2['belowThis'] = ((df2['rank'] / mMax) * 100).round(2)
df3 = df2[['COV', 'belowThis']].sort_values('COV', ascending=True)
#print(df3)

df3g = df3[df3['belowThis'] <= 30]
df3r = df3[df3['belowThis'] >= 85]

fig, ax = plt.subplots()
ax.plot(df3['belowThis'], df3['COV'], 'k-', label='COV')
ax.fill_between(df3g['belowThis'], 0, df3g['COV'], label='0-30%', facecolor='green')
ax.fill_between(df3r['belowThis'], 0, df3r['COV'], label='85-100%', facecolor='red')

plt.ylim(0, 0.1)
plt.xlim(0, 100)
plt.legend(loc='upper left')
plt.show()

