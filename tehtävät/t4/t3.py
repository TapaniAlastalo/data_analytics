import pandas as pd

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/lam.csv', sep = ',', decimal='.')
df.fillna(0, inplace=True)
df['Päivä'] = df['Pvm'].str.split('.').str[0]
df2 = df.pivot_table(['määrä'], index=['tuntiväli'], columns=['Päivä'], aggfunc=sum).astype(int)
df2['suurin'] = df2.max(axis=1)
df2['sija'] = df2.iloc[:, 0:30].rank(axis=1, ascending=False)['määrä'].iloc[:, 0]
print(df2)