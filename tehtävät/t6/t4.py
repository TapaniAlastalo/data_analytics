import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/km.txt', sep=",", decimal='.')#, index_col='pvm')

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

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/km.txt', 
                 parse_dates=['pvm'], 
                 dayfirst=True, 
                 index_col='pvm')

#alkupeäriset tiedot vuodelta 2019, toimii koska pvm sarake on datetime muodossa
df2019A = df['2019']

#luodaan haluttu aikaväli
days2019 = pd.date_range('2019-05-01', '2019-10-31') 

#luodaan aikavälistä uusi dataframe ja otetaan sinne alkuperäiset kilometrit ja muokataan NaN arvot nolliksi
df2019F = days2019.to_frame()
df2019F['km'] = df2019A['km']
df2019F['km'].fillna(0, inplace=True)
df2019F.drop(0, axis=1, inplace=True) #poistetaan turha päivämärää sarake
df2019F['yht'] = df2019F['km'].cumsum() #lasketaan kilsat alusta alkaen yhteen
df2019F.reset_index(inplace=True) #nollataan indeksi jotta seuraavien vuosien taulussa on sama indeksi laskuja varten
#print(df2019F.head())

#toistetaan samat toimet kaikille vuosille, kenties olisin voinut hyödyntää Period toimintoa mutta en saanut sen avulla toimimaan
df2018A = df['2018']
days2018 = pd.date_range('2018-05-01', '2018-10-31')
df2018F = days2018.to_frame()
df2018F['km'] = df2018A['km']
df2018F['km'].fillna(0, inplace=True)
df2018F.drop(0, axis=1, inplace=True)
df2018F['yht'] = df2018F['km'].cumsum()
df2018F.reset_index(inplace=True)
df2018F['erotus'] = df2019F['yht'] - df2018F['yht']
#print(df2018F.head())

df2017A = df['2017']
days2017 = pd.date_range('2017-05-01', '2017-10-31')
df2017F = days2017.to_frame()
df2017F['km'] = df2017A['km']
df2017F['km'].fillna(0, inplace=True)
df2017F.drop(0, axis=1, inplace=True)
df2017F['yht'] = df2017F['km'].cumsum()
df2017F.reset_index(inplace=True)
df2017F['erotus'] = df2019F['yht'] - df2017F['yht']
#print(df2017F.head())

df2016A = df['2016']
days2016 = pd.date_range('2016-05-01', '2016-10-31')
df2016F = days2016.to_frame()
df2016F['km'] = df2016A['km']
df2016F['km'].fillna(0, inplace=True)
df2016F.drop(0, axis=1, inplace=True)
df2016F['yht'] = df2016F['km'].cumsum()
df2016F.reset_index(inplace=True)
df2016F['erotus'] = df2019F['yht'] - df2016F['yht']
#print(df2016F.head())

df2015A = df['2015']
days2015 = pd.date_range('2015-05-01', '2015-10-31')
df2015F = days2015.to_frame()
df2015F['km'] = df2015A['km']
df2015F['km'].fillna(0, inplace=True)
df2015F.drop(0, axis=1, inplace=True)
df2015F['yht'] = df2015F['km'].cumsum()
df2015F.reset_index(inplace=True)
df2015F['erotus'] = df2019F['yht'] - df2015F['yht']
#print(df2015F.head())

#luodaan tiedoista kaavio
plt.figure()
plt.plot(df2019F['index'], df2018F['erotus'], label='ero2018')
plt.plot(df2019F['index'], df2017F['erotus'], label='ero2017')
plt.plot(df2019F['index'], df2016F['erotus'], label='ero2016')
plt.plot(df2019F['index'], df2015F['erotus'], label='ero2015')
fig = plt.gcf()
fig.set_size_inches(20,8) #muokataan kaavion kokoa
plt.legend()
plt.show()

