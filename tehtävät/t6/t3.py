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
#df['Prev15'] = df['JYPin Pisteet'].shift(1, fill_value=0) + df['JYPin Pisteet'].shift(2, fill_value=0) + df['JYPin Pisteet'].shift(3, fill_value=0) + df['JYPin Pisteet'].shift(4, fill_value=0) + df['JYPin Pisteet'].shift(5, fill_value=0) + df['JYPin Pisteet'].shift(6, fill_value=0) + df['JYPin Pisteet'].shift(7, fill_value=0) + df['JYPin Pisteet'].shift(8, fill_value=0) + df['JYPin Pisteet'].shift(9, fill_value=0) + df['JYPin Pisteet'].shift(10, fill_value=0) + df['JYPin Pisteet'].shift(11, fill_value=0) + df['JYPin Pisteet'].shift(12, fill_value=0) + df['JYPin Pisteet'].shift(13, fill_value=0) + df['JYPin Pisteet'].shift(14, fill_value=0) + df['JYPin Pisteet'].shift(15, fill_value=0)
#print(df)

df2 = df[['pvm', 'JYPin Pisteet']]
df2.set_index(['pvm'],inplace=True)
#print(df2)
df2['JYPin Pisteet'].rolling(15).sum().plot(style='r:o', markersize=3, label='Edelliset 15 peliä')
plt.legend()
plt.show()