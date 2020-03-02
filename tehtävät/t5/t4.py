import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta5/saa_jkl20200229.csv', sep = ',', decimal='.')
df['Lumensyvyys (cm)'].replace({-1: 0}, inplace=True)
df = df[(df['Kk'] <= 2) & (df['Vuosi'] >= 2018)]

def laskeMoneskoPv(x):
    pv = 0
    if(x['Kk'] == 2):
        pv += 31    
    return x['Pv'] + pv

df['vuodenpv'] = df.apply(laskeMoneskoPv, axis=1)
print(df)

df2 = df.groupby(['Vuosi','vuodenpv'])['Lumensyvyys (cm)'].max().unstack()
print(df2['Vuosi'])

fig, ax = plt.subplots()

#plt.plot('Vuosi', 'Lumensyvyys (cm)', 'r-', label='jee', data = df2)
#plt.plot('Lumensyvyys (cm)', data = df2)
#plt.plot(df['Vuosi'],df.iloc[:,1:])
#df.plot('Lumensyvyys (cm)',['2018', '2019', '2020'])
#plt.legend(loc='upper left')
#plt.show()