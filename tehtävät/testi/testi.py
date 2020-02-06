import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta2/nhl1.csv', sep = ',')
#dfT = df.T
print('Testi')
print(df)

# otsikot uusiksi
tulostettavat = ['Player', 'Team', 'Birth City', 'Ntnlty', 'Ht', 'Wt', 'Overall', 'GP', 'G']
tulostettavatSuomeksi = ['Pelaaja', 'Joukkue', 'Syntymakaupunki', 'Kansalaisuus', 'Pituus', 'Paino', 'Varausnro', 'Pelit', 'Maalit']

print('or_______-')
df2 = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta2/nhl1.csv', sep=',', usecols=tulostettavat, names=tulostettavatSuomeksi)
print(df2)
print('___end or___EI VAAN TOIMI, VAIKKA PITaIS. JOKU '' VOI OLLA VaaRa. JOSKUS SAIN TOIMII, mutta ei kokonaisena')


df[tulostettavatSuomeksi]=df[tulostettavat]
# muunnokset
#df['Pituus'] = round((df['Ht'] * 2.54), 1)
#df['Paino'] = round((df['Wt'] * 0.453592), 1)#("%.1f " % (df['Wt'] * 0.453592))

#print(df[tulostettavat])
#print(df[tulostettavatSuomeksi])

# painoluokka
print('Painoluokat')
df['Painoluokka'] = df['Paino'] / 5
tulostettavatSuomeksi.append('Painoluokka')
print(df[tulostettavatSuomeksi])
#print(df)

#print('Jyvaskylalaiset')
#jyp = df[(df['Syntymakaupunki']=='Jyvaskyla')]
#print(jyp)