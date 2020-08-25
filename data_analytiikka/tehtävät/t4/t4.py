import pandas as pd

#df = pd.read_csv('http://gpspekka.kapsi.fi/trafi56.zip', sep = ',', decimal=',', encoding='latin_1', nrows=10000)
df = pd.read_csv('../testi/trafi56.zip', sep = ';', decimal=',', usecols=['kayttoonottopvm', 'merkkiSelvakielinen', 'ensirekisterointipvm', 'sahkohybridi'], encoding='latin_1', nrows=4000000)
#df = pd.read_csv('../testi/trafi56.zip', sep = ';', decimal=',', usecols=['kayttoonottopvm', 'merkkiSelvakielinen', 'ensirekisterointipvm', 'sahkohybridi'], encoding='latin_1')
#df['time'] = %%time

df2 = df
#print(df2.iloc[:,0:10])


print('\nkuinka monta (vielä liikennekäytössä eli tiedostossa olevaa) ajoneuvoa käyttöönotettiin (kayttoonottopvm) 18-vuotissyntymäpäivänäsi')
dft1 = df2['kayttoonottopvm'] == 20010406
print(dft1.sum())

print('\nmikä on viiden kärki kentän merkkiSelvakielinen lukumäärissä')
dft2 = pd.DataFrame(df2.groupby('merkkiSelvakielinen')['merkkiSelvakielinen'].count().sort_values(ascending=False))
print(dft2.head(5))

print('\ntee uusi sarake rekvuosi, johon luet 4 ensimmäistä merkkiä kentästä ensirekisterointipvm. Ristiintaulukoi sitten kentän sahkohybridi prosenttiosuudet (True/False) rekvuoden mukaan vuodesta 2001 eteenpäin.')
df2['rekvuosi'] = 0
df2.loc[(df['ensirekisterointipvm'].notnull()), 'rekvuosi'] = df2['ensirekisterointipvm'].str[0:4].astype(float)
dft3 = df2[(df2['rekvuosi'] > 2000)]
dft3 = pd.crosstab(dft3['rekvuosi'],  dft3['sahkohybridi'], normalize = 'index').applymap("{:.1%}".format)
print(dft3)