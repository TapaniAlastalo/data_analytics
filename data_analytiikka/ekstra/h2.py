import pandas as pd
import numpy as np

df = pd.read_csv('https://student.labranet.jamk.fi/~huoptu/datananal_s20/kerta2/harjoitukset/ehi_010_2020q2.csv', sep=",", decimal='.')
print(df)

columns = ['Year Total €/MWh']
#a.value[a['words'].isin(b)] = 0
b = ['.']
#df.values[df[columns[0]].isin(b)] = 0 #np.NAN
#df[df[columns[0]].isin(['.'])]
print(df[df[columns[0]].isin(b)])
#print(df)

#data = df[columns]
#print(data)

#cutted_data = pd.cut(data, 5)
#print(cutted_data)


#df['pvm']  = pd.to_datetime(df['pvm'])
#df.set_index('pvm', inplace=True)
#df.sort_index(ascending=True, inplace=True)
#print(df)
#df['2019']
#df2019 = df['2019-05-01':'2019-10'].resample('1D').asfreq().fillna(0)#.ffill()
