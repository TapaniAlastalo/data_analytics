import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta3/kunnat.txt', sep = ';', decimal=',')

print("a) Seutukuntien pinta-alat:")
df1 = pd.DataFrame(df['maapinta-ala'].groupby(df['seutukunta']).sum())
print(df1.sort_values('maapinta-ala', ascending=False).head(5))

print("b) maakuntien kaupungistuminen:")
df['kaupunkilaiset'] =  0
df.loc[(df['kuntamuoto'] == "Kaupunki"), 'kaupunkilaiset'] = df['Väkiluku']

kaupunkilaiset = pd.DataFrame((df['kaupunkilaiset'].groupby(df['maakunta']).sum()) / (df['Väkiluku'].groupby(df['maakunta']).sum()) * 100)
kaupunkilaiset['kaupungistuminen%'] = kaupunkilaiset.iloc[:, 0]
print(kaupunkilaiset.sort_values(['kaupungistuminen%'], ascending=False).head(5))

print("c) Ruotsinkielisten osuus maakunnittain")
df['ruottalaiset'] =  df['Ruotsinkielisten osuus%'] * df['Väkiluku'] / 100
ruottalaiset = pd.DataFrame((df['ruottalaiset'].groupby(df['maakunta']).sum()) / (df['Väkiluku'].groupby(df['maakunta']).sum()) * 100) 
ruottalaiset['ruottalaiset%'] = ruottalaiset.iloc[:, 0]
print(ruottalaiset.sort_values('ruottalaiset%', ascending=False).head(5))