import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/epl4.txt', sep = ',', decimal='.')

print(df)

df2 = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/sijat.txt', sep = ',', decimal='.')

print('test')
#te = df.pivot_table(['HomePoints', 'AwayPoints'], index=['Season'])
#print(te)

print('laske jokaiselle joukkueelle jokaiselle kaudelle kuinka paljon pisteitä on kauden 15 ensimmäisestä pelistä')
#t1 = pd.crosstab([df['HomeTeam'], df['AwayTeam']],  [df['HomePoints'], df['AwayPoints']])
#print(t1)

k3 = df[df['HomePoints']==3].groupby(['Season', 'HomeTeam'])['HomeTeam'].count()  # 3p-kotivoitot
k1 = df[df['HomePoints']==1].groupby(['Season', 'HomeTeam'])['HomeTeam'].count()  # 1p-kotona
k0 = df[df['HomePoints']==0].groupby(['Season', 'HomeTeam'])['HomeTeam'].count()  # 0p-kotitappiot

v3 = df[df['AwayPoints']==3].groupby(['Season', 'AwayTeam'])['AwayTeam'].count()  # 3p-vierasvoitot
v1 =  df[df['AwayPoints']==1].groupby(['Season', 'AwayTeam'])['AwayTeam'].count()  # 1p-vieraana
v0 = df[df['AwayPoints']==0].groupby(['Season', 'AwayTeam'])['AwayTeam'].count()  # 0p-vierastappiot

# Seriekset DataFrameksi
t = pd.DataFrame({'k3': k3, 'v3': v3, 'k1':k1, 'v1':v1, 'k0':k0, 'v0':v0})
print(t)
t['o'] = (t['k3'] + t['v3']) + (t['k1'] + t['v1']) + (t['k0'] + t['v0'])
t['p'] = ((t['k3'] + t['v3']) *3) + (t['k1'] + t['v1'])
print(t)

print('laske jokaiselle kaudelle paljonko on neljänneksi suurin pistemäärä (15 ensimmäisen pelin pistemääristä)')

st = t['p'].unstack()
print(st)
sst =st.unstack()
print(sst)
#t['rank'] = t['p'].rank(axis=0, ascending=False)
#print(t.iloc[:, 0])

print('laske jokaiselle joukkueelle jokaiselle kaudelle paljonko ero on ollut tähän neljänteen pistemäärään')

print('suodata mukaan vain ne joukkueet, jotka ko. kaudella ovat loppusijoituksissa 4 parhaan joukossa ja laijittele näistä eroista suurimmat (eniten 4 sijaa jäljessä olleet).')