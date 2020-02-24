import pandas as pd

#df = pd.read_csv('http://gpspekka.kapsi.fi/trafi56.zip', sep = ',', decimal=',', encoding='latin_1', nrows=10000)
df = pd.read_csv('../testi/trafi56.zip', sep = ';', decimal=',', encoding='latin_1', nrows=10000)
#df['time'] = %%time

print(df)
df1 = df.iloc[:,0:10]
print(df1)

print('\nkuinka monta (vielä liikennekäytössä eli tiedostossa olevaa) ajoneuvoa käyttöönotettiin (kayttoonottopvm) 18-vuotissyntymäpäivänäsi')
dft1 = df['kayttoonottopvm'] == 20010406
print(dft1.sum())

print('\nmikä on viiden kärki kentän merkkiSelvakielinen lukumäärissä')
dft2 = pd.DataFrame(df.groupby('merkkiSelvakielinen')['merkkiSelvakielinen'].count().sort_values(ascending=False))
print(dft2.head(5))

print('\ntee uusi sarake rekvuosi, johon luet 4 ensimmäistä merkkiä kentästä ensirekisterointipvm. Ristiintaulukoi sitten kentän sahkohybridi prosenttiosuudet (True/False) rekvuoden mukaan vuodesta 2001 eteenpäin.')
#ensirekisterointipvm
#sahkohybridi
#ensirekisterointipvm
df['rekvuosi'] = 0
df.loc[(df['ensirekisterointipvm'].notnull()), 'rekvuosi'] = df['ensirekisterointipvm'].str.split('-').str[0]
print('---------')
#print(df['rekvuosi'])
#print(df['sahkohybridi'].sort_values(ascending=False))

print('00000000')
#dft31 = df[(['rekvuosi']) df['rekvuosi'].astype(int) >= 2001]
#print(dft31.head(5))

print("dft3v1")
dft3v1 = pd.crosstab(df['rekvuosi'],  df['sahkohybridi'])
print(dft3v1)

#print("dft3v2")
#dft3v2 = df[(df['rekvuosi'].astype(int) >= 2001), df['sahkohybridi'].groupby(df['rekvuosi']).count()]
#dft32['rekvuosi'] = dft32.iloc[:, 0]    #.apply(''.join)
#dft3v2 = dft3v2['rekvuosi'].astype(int) >= 2001
#print(dft3v2.head(7))

print("dft3v4")
dft3v3 = df.pivot_table(columns=['sahkohybridi'], index=['rekvuosi'])
print(dft3v3)

print("dft3v4")
dft3v4 = df[(df['rekvuosi'].astype(int) >= 2001), 'rekvuosi'].groupby('sahkohybridi')   #['Pukin maalit'].count().sort_values(ascending=False)
print(dft3v4)