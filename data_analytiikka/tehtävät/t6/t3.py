import pandas as pd
import datetime
import matplotlib.pyplot as plt

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal1k2020/kerta6/liiga.txt', sep=",", decimal='.')
df['KotiJ'] = df['ottelu'].str.split('-').str.get(0).str.strip()
df['VierasJ'] = df['ottelu'].str.split('-').str.get(1).str.strip()

jyp = 'JYP'
df = df[(df['KotiJ']==jyp) | (df['VierasJ']==jyp)]

df['KotiM'] = df['tulos'].str.split('-').str.get(0).str.strip().astype(int)
df['VierasM'] = df['tulos'].str.split('-').str.get(1).str.strip().astype(int)

df['KotiP'] = 0  # "alustetaan" pisteet nolliksi
df['VierasP'] = 0

df.loc[(df['huom'].isnull()) & (df['KotiM']>df['VierasM']), 'KotiP'] = 3 # kotivoitosta 3 pistettä jos NaN
df.loc[(df['huom'].notnull()) & (df['KotiM']>df['VierasM']), 'KotiP'] = 2 # kotivoitosta 2 pistettä jos JA/VL
df.loc[(df['huom'].notnull()) & (df['KotiM']<df['VierasM']), 'KotiP'] = 1 # kotitappiosta 1 piste jos JA/VL

df.loc[(df['huom'].isnull()) & (df['KotiM']<df['VierasM']), 'VierasP'] = 3 # vierasvoitosta 3 pistettä jos NaN
df.loc[(df['huom'].notnull()) & (df['KotiM']<df['VierasM']), 'VierasP'] = 2 # vierasvoitosta 2 pistettä jos JA/VL
df.loc[(df['huom'].notnull()) & (df['KotiM']>df['VierasM']), 'VierasP'] = 1 # vierastappiosta 1 piste jos JA/VL

df['JYPin Pisteet'] = 0
df.loc[(df['KotiJ']==jyp), 'JYPin Pisteet'] = df['KotiP']
df.loc[(df['VierasJ']==jyp), 'JYPin Pisteet'] = df['VierasP']
#print(df)

df2 = df[['pvm', 'JYPin Pisteet']]
df2['pvm'] = pd.to_datetime(df2['pvm'], dayfirst=True )
df2.set_index(['pvm'],inplace=True)

plt.figure()
plt.plot(df2['JYPin Pisteet'].rolling(15).sum(), 'r:o', markersize=3, label='Edelliset 15 peliä')
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.legend(loc='lower left')
plt.show()