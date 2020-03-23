import pandas as pd
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/accesslog.zip', sep=",", decimal='.')
#print(df)
#df['date'], df['h'], df['min'], df['s'] = df['timestamp'].str.split(':', 3).str
#df['y'], df['m'], df['d'] = df['date'].str.split('/', 2).str
df['dt'] = df['timestamp'].str.replace(':', ' ', 1)
print(df)
df['timestamp'] = df['dt'].map(lambda x: parse(x))
print(df['timestamp'])
#df['pvm']  = pd.to_datetime(df['timestamp'])
#df  = pd.to.datetime(df['timestamp'], format='%d:%b/%Y:%H:%M:%S')
#df.set_index('pvm', inplace=True)
#df.sort_index(ascending=True, inplace=True)
#print(df)
df['timestamp']  = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df.sort_index(ascending=True, inplace=True)
print(df)



df2 = df['2018-05-08 11:00:00':'2018-05-08 14:00:00']
print(df2)

#plt.plot(df2019['vuodenpv'], df2019['Lumensyvyys (cm)'], 'r-', label='2019')
df['eventid'].count.plot()
#df['km'].rolling('7d').mean().plot(label='rollWeek')
#df['km'].rolling('365d').mean().plot(label='rollYear')
plt.legend()

plt.show()



