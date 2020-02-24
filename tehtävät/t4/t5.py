import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/epl4.txt', sep = ',', decimal='.')

print(df)

df2 = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/sijat.txt', sep = ',', decimal='.')

print(df2)

print('laske jokaiselle joukkueelle jokaiselle kaudelle kuinka paljon pisteitä on kauden 15 ensimmäisestä pelistä')
t1 = pd.crosstab([df['HomeTeam'], df['AwayTeam']],  [df['HomePoints'], df['AwayPoints']])
print(t1)


print('laske jokaiselle kaudelle paljonko on neljänneksi suurin pistemäärä (15 ensimmäisen pelin pistemääristä)')

print('laske jokaiselle joukkueelle jokaiselle kaudelle paljonko ero on ollut tähän neljänteen pistemäärään')

print('suodata mukaan vain ne joukkueet, jotka ko. kaudella ovat loppusijoituksissa 4 parhaan joukossa ja laijittele näistä eroista suurimmat (eniten 4 sijaa jäljessä olleet).')