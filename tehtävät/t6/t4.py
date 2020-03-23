import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/km.txt', sep=",", decimal='.')#, index_col='pvm')

#df['Vuosi'] = df['pvm'].str.split('-').str.get(0).str.strip().astype(int)
#df['Kk'] = df['pvm'].str.split('-').str.get(1).str.strip().astype(int)
#df['Pv'] = df['pvm'].str.split('-').str.get(2).str.strip().astype(int)
df['y'], df['m'], df['d'] = df['pvm'].str.split('-', 2).str

df['pvm']  = pd.to_datetime(df['pvm'])
df.set_index('pvm', inplace=True)
df.sort_index(ascending=True, inplace=True)
print(df)

print(df['2019'])
print(df[(df['y']==2019)]['km'])
#print(df['Kk'].sort_values(ascending=False).head(10))

#plt.figure()  
#ax1 = plt.subplot(2,2,1) 
#df[df['2019']].plot(y='Lumensyvyys (cm)', label='2019', ax=ax1)
#df[df['2018']].plot(y='Lumensyvyys (cm)', label='2018', ax=ax1)
#df[df['2017']].plot(y='Lumensyvyys (cm)', label='2017', ax=ax1)
#plt.legend()
#plt.ylabel('lumensyvyydet tammikuussa')

#fig = plt.gcf()  #antaa nykyisen kuvion (current figure)
#fig.set_size_inches(14, 14)

#plt.show()

df['km'].plot()
df['km'].rolling('7d').mean().plot(label='rollWeek')
df['km'].rolling('365d').mean().plot(label='rollYear')
plt.legend()

plt.show()

#df['km'].plot()
#df['km'].rolling(20).mean.plot(label='1d')
#df['km'].rolling(20, center=True).mean().plot(label='roll20c')
#plt.legend()
#plt.show()