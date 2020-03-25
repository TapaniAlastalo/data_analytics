import pandas as pd
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/accesslog.zip', sep=",", decimal='.', nrows=10000)
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


print("--------df2----------")
df2 = df['2018-05-08 11:00:00':'2018-05-08 14:00:00']
#df2['time'] = 
print(df2)
df3 = df2.groupby(['timestamp','eventid'])['ip']#.nunique().sort_values(ascending=False)#.unstack().count()
#df3.fillna(0, inplace=True)
print(df3)

#df22 = df2.resample('5min')
#print(df22)

#event1 = df3[df3['eventid'] == 2018]
#df4 = df3.unstack().fillna(0)
#print(df4)
#print(df3.unstack().unstack())
#print('test')
#print(df4.resample('5min').count())
#print('testeest')
#print(df4['20180508_MQ_MA'])
#print(df4['20180508_MQ_MA'].resample('5min').count())

#df4['20180508_MQ_MA'].rolling(300).count().plot()
#df4['20180508_MQ_MA'].rolling('5min').mean().plot()
#df4['20180508_MQ_MB'].rolling('5min').mean().plot()
#df4['20180508_MQ_MC'].rolling('5min').sum().plot()
#df4['20180508_MQ_WA'].rolling('5min').count().plot()
#df4['20180508_MQ_WB'].rolling('5min').count().plot()
#df4['20180508_MQ_WC'].rolling('5min').count().plot()
#plt.legend()
#plt.show()