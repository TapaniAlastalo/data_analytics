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
#print(df)

df2 = df.groupby(['Vuosi','vuodenpv'])['Lumensyvyys (cm)', 'Ilman lämpötila (degC)'].max()#.unstack()
print(df2)
print(df2.unstack())

print('reset')
df2 = df2.reset_index()
print(df2)

df2018 = df2[df2['Vuosi'] == 2018]
df2019 = df2[df2['Vuosi'] == 2019]
df2020 = df2[df2['Vuosi'] == 2020]

df2020['muutoseiliseen'] = df2020['Lumensyvyys (cm)'] - df2020['Lumensyvyys (cm)'].shift()
df2020.fillna(0, inplace=True)
df2020 = df2020.reset_index()
df2019 = df2019.reset_index()
df2020['erotus2019'] = df2020['Lumensyvyys (cm)'] - df2019['Lumensyvyys (cm)']
df2020.fillna(0, inplace=True)

#fig, (ax1, ax2, ax3, ax4) = plt.subplots() #(2, 2, figsize=(14, 14))
#fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(14, 14))

#ax1.plot(df2018['vuodenpv'], df2018['Lumensyvyys (cm)'], 'b-', label='2018')
#ax1.plot(df2019['vuodenpv'], df2019['Lumensyvyys (cm)'], 'r-', label='2019')
#ax1.plot(df2020['vuodenpv'], df2020['Lumensyvyys (cm)'], 'g-', label='2020')
#ax1.set_xlim(0, 60)

#ax2.plot(df2018['vuodenpv'], df2018['Lumensyvyys (cm)'], 'b-', label='2018')
#ax3.plot(df2019['vuodenpv'], df2019['Lumensyvyys (cm)'], 'r-', label='2019')
#ax4.plot(df2020['vuodenpv'], df2020['Lumensyvyys (cm)'], 'g-', label='2020')

#plt.legend(loc='upper left')

plt.figure(figsize=(14, 14))  # luodaan uusi kuvio, tämä on "nykyinen kuvio".

plt.subplot(2,2,1)  # tehdään nykyiseen kuvioon 2 riviä, 1 sarake -"ruudukko" ja otetaan 1. paikka "nykyiseksi kaavioksi"
plt.plot(df2018['vuodenpv'], df2018['Lumensyvyys (cm)'], 'b-', label='2018')
plt.plot(df2019['vuodenpv'], df2019['Lumensyvyys (cm)'], 'r-', label='2019')
plt.plot(df2020['vuodenpv'], df2020['Lumensyvyys (cm)'], 'g-', label='2020')
plt.legend(loc='upper left') # lisätään selite "nykyiseen kaavioon"
plt.xlim(0, 60)

plt.subplot(2,2,2)
plt.plot(df2018['vuodenpv'], df2018['Lumensyvyys (cm)'], 'b-', label='2018')
plt.plot(df2019['vuodenpv'], df2019['Lumensyvyys (cm)'], 'r-', label='2019')
plt.plot(df2020['vuodenpv'], df2020['Lumensyvyys (cm)'], 'g-', label='2020')
plt.xlim(0, 60)

plt.subplot(2,2,3)
#plt.bar(df2020['muutoseiliseen'], df2020['vuodenpv'])
plt.hist(df2020['muutoseiliseen'], df2020['vuodenpv'])
plt.xlim(-15, 15)
#plt.ylim(0, 75)


#plt.subplot(2,2,4)
#plt.plot(df2018['vuodenpv'], df2018['Lumensyvyys (cm)'], 'b-', label='2018')
#plt.plot(df2019['vuodenpv'], df2019['Lumensyvyys (cm)'], 'r-', label='2019')
#plt.plot(df2020['vuodenpv'], df2020['Lumensyvyys (cm)'], 'g-', label='2020')
#plt.xlim(0, 60)
#plt.figure(2)
#sns.regplot(df2020['Ilman lämpötila (degC)'], df2020['muutoseiliseen'])

#plt.show()