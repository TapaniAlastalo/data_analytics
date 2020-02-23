import pandas as pd

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/lam.csv', sep = ',', decimal='.')
# korvataan puuttuvat arvot nollilla
df.fillna(0, inplace=True)
df['Päivä'] = df['Pvm'].str.split('.').str[0]
#df['mrä'] = df['määrä'].astype(int)

df2 = df.pivot_table(['määrä'], index=['tuntiväli'], columns=['Päivä'], aggfunc=sum, margins=False).applymap('{:,.0f}'.format)
df2['suurin'] = df2.max(axis=1)
#df2['suurin'] = df2.max(level='tuntiväli')
#df2['suurin'] = df2.max(axis=1, level=None, numeric_only=None)

m = df2.max(axis=1)
#n = df2.idxmax(axis=1)
df2 = df2.sort_values(['suurin'])
m['sija'] = range(24)
print(df2)
print(m)
#print(n)