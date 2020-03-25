import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/km.txt', sep=",", decimal='.')#, index_col='pvm')

#df['Vuosi'] = df['pvm'].str.split('-').str.get(0).str.strip().astype(int)
#df['Kk'] = df['pvm'].str.split('-').str.get(1).str.strip().astype(int)
#df['Pv'] = df['pvm'].str.split('-').str.get(2).str.strip().astype(int)
#df['end'] = df['pvm'].str.split('-').str.get(1).str.strip() + df['pvm'].str.split('-').str.get(2).str.strip()
#df['y'], df['m'], df['d'] = df['pvm'].str.split('-', 2).str

df['pvm']  = pd.to_datetime(df['pvm'])
df.set_index('pvm', inplace=True)
df.sort_index(ascending=True, inplace=True)
print(df)
#df['2019']
df2019 = df['2019-05-01':'2019-10'].resample('1D').asfreq().fillna(0)#.ffill()
df2018 = df['2018-05-01':'2018-10'].resample('1D').asfreq().fillna(0)
df2017 = df['2017-05-01':'2017-10'].resample('1D').asfreq().fillna(0)
df2016 = df['2016-05-01':'2016-10'].resample('1D').asfreq().fillna(0)
df2015 = df['2015-05-01':'2015-10'].resample('1D').asfreq().fillna(0)
#df2018.set_index('end', inplace = True)
#df2019.set_index('end', inplace = True)
df2019 = df2019.reset_index()
df2018 = df2018.reset_index()
df2017 = df2017.reset_index()
df2016 = df2016.reset_index()
df2015 = df2015.reset_index()

df2019['sum'] = df2019['km'].cumsum()
df2018['sum'] = df2018['km'].cumsum()
df2017['sum'] = df2017['km'].cumsum()
df2016['sum'] = df2016['km'].cumsum()
df2015['sum'] = df2015['km'].cumsum()

print(df2018)
print(df2019)
#print(df2019b)
print('test')
df2018['erotus'] = df2018['sum'] - df2019['sum']
df2017['erotus'] = df2017['sum'] - df2019['sum']
df2016['erotus'] = df2016['sum'] - df2019['sum']
df2015['erotus'] = df2015['sum'] - df2019['sum']
print(df2018)
#df2018roll = df2018['km'].rolling('1d').sum() - df2019['km'].rolling('1d').sum()
#print(df2018roll)
#df2018['km'].rolling(1).sum().plot(label='roll1')
#df2019['km'].rolling(1, center=True).sum().plot(label='roll2')
#(df2018['km'].rolling('1d').sum()).plot(label='roll3')

df2018['erotus'].plot(label='2018')
df2016['erotus'].plot(label='2017')
df2015['erotus'].plot(label='2016')
df2015['erotus'].plot(label='2015')
plt.legend()
plt.show()
