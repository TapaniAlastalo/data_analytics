import pandas as pd
import datetime as dt

names=['id-aika', 'longitudi', 'latitudi', 'nopeus']
df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/gps.txt', sep="_", decimal=',', names=names)

#df['id'], df['aikaero'] = df['id-aika'].str.split('.', 1).str
df['id'] = df['id-aika'].str.split('.').str.get(0)
df['aikaero'] = df['id-aika'].str.split('.').str.get(1).astype(int)

alkuaika = dt.datetime(2006,1,1,0,0,0)
df['aika'] = alkuaika + pd.to_timedelta(df['aikaero'], 's')

df['longitudi'] = df['longitudi'] / 50000
df['latitudi'] = df['latitudi'] / 100000
df['nopeus'] = df['nopeus'].str[1:].astype(float) / 10

newnames=['id', 'aika', 'longitudi', 'latitudi', 'nopeus']
df2 = df[newnames]
print(df2)

