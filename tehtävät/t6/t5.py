import pandas as pd
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
#import seaborn as sns

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/accesslog.zip', sep=",", decimal='.')#, nrows=10000)

#df['date'], df['h'], df['min'], df['s'] = df['timestamp'].str.split(':', 3).str
#df  = pd.to.datetime(df['timestamp'], format='%d:%b/%Y:%H:%M:%S')
df['dt'] = df['timestamp'].str.replace(':', ' ', 1)
df['timestamp'] = df['dt'].map(lambda x: parse(x))
df['timestamp']  = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df['2018-05-08 10:50:00':'2018-05-08 14:00:00']
df.sort_index(ascending=True, inplace=True)
#print(df)

df2 = pd.DataFrame(df.groupby('eventid').resample('5min')['ip'].nunique())
df2.reset_index(inplace=True)
#print(df2)

dfMA = df2.loc[(df2['eventid']=='20180508_MQ_MA')]
dfMA.set_index('timestamp', inplace=True)
#print(dfMA)
dfMB = df2.loc[(df2['eventid']=='20180508_MQ_MB')]
dfMB.set_index('timestamp', inplace=True)
dfMC = df2.loc[(df2['eventid']=='20180508_MQ_MC')]
dfMC.set_index('timestamp', inplace=True)
dfWA = df2.loc[(df2['eventid']=='20180508_MQ_WA')]
dfWA.set_index('timestamp', inplace=True)
dfWB = df2.loc[(df2['eventid']=='20180508_MQ_WB')]
dfWB.set_index('timestamp', inplace=True)
dfWC = df2.loc[(df2['eventid']=='20180508_MQ_WC')]
dfWC.set_index('timestamp', inplace=True)

plt.figure()
plt.plot(dfMA['ip'], label='20180508_MQ_MA')
plt.plot(dfMB['ip'], label='20180508_MQ_MB')
plt.plot(dfMC['ip'], label='20180508_MQ_MC')
plt.plot(dfWA['ip'], label='20180508_MQ_WA')
plt.plot(dfWB['ip'], label='20180508_MQ_WB')
plt.plot(dfWC['ip'], label='20180508_MQ_WC')

fig = plt.gcf()
fig.set_size_inches(10,6)
xMin = datetime.datetime(2018,5,8,10,55,0)
xMax = datetime.datetime(2018,5,8,14,0,0)
plt.xlim(xMin, xMax)
plt.xlabel('timestamp')
plt.ylabel('ip-osoitteet')
plt.legend()
plt.show()