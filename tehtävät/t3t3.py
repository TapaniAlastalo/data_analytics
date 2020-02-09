import pandas as pd

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta3/kone.csv', sep = ',')

df['katko_alkaa'] = (df['katko']>0) & (df['katko'].shift(1)==0)  
df['katko_nro']  = df['katko_alkaa'].cumsum()

katkot = pd.DataFrame(df['katko_nro'][df['katko_nro']>0].groupby(df['katko_nro']).first())
katkot['Alkoi'] = df['aika'].groupby(df['katko_nro']).first()
katkot['Loppui'] = df['aika'].groupby(df['katko_nro']).last()
katkot['kesto'] = (pd.to_datetime(katkot['Loppui'])) - (pd.to_datetime(katkot['Alkoi']))

print(katkot.head(10))
