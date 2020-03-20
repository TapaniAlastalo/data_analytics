import pandas as pd
import datetime

names=['id-aika', 'longitudi', 'latitudi', 'nopeus']
df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/epl20200309.txt', sep=";", decimal='.')

df['Date']  = pd.to_datetime(df['Date'])
print(df)

kkeniten = df.resample('M',on='Date', kind='period').size().sort_values(ascending=False).head(10)
print(kkeniten)

#viikoina ma-su
wkeniten = df.resample('W',on='Date', kind='period', label='left').size().sort_values(ascending=False).head(10)
print(wkeniten)

#vkonpäivinä
df['DoW'] = df['Date'].dt.dayofweek
df['Day'] = df['Date'].dt.day_name()

#test = df['DoW'].groupby([df['DoW']]).count()
#print(test)
pveniten = df.groupby('DoW')['DoW'].count()
print(pveniten)

#best = df['DoW'].dt.day_name
#print(best)