import pandas as pd

#df = pd.read_csv('http://gpspekka.kapsi.fi/trafi56.zip', sep = ',', decimal=',', encoding='latin_1', nrows=10000)
df = pd.read_csv('../testi/trafi56.zip', sep = ';', decimal=',', usecols=['kayttoonottopvm', 'merkkiSelvakielinen', 'ensirekisterointipvm', 'sahkohybridi'], encoding='latin_1', nrows=100000)
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
df2['rekvuosi'] = None
df2.loc[(df['ensirekisterointipvm'].notnull()), 'rekvuosi'] = df2['ensirekisterointipvm'].str[0:4]
dft3v1 = pd.crosstab(df2['rekvuosi'],  df2['sahkohybridi'], normalize = 'index').applymap("{:.1%}".format)
print(dft3v1)