# Tehtävän toteutus

import pandas as pd
# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df = pd.read_csv('c:/data/lampotila_2010.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df = df.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
#df = df.set_index('date')
df = df[['date','Temperature']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_temps_2010 = df

df = pd.read_csv('c:/data/sahkonkulutus_2010.csv')
df = df.rename(columns={'Alkuaika UTC':'date','Sähkön kulutus Suomessa':'Electricity Consumption'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
#df = df.set_index('date')
df = df[['date','Electricity Consumption']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_elec_2010 = df


# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df = pd.read_csv('c:/data/lampotila_2011.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df = df.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
df = df.set_index('date')
df = df[['Temperature']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_temps_2011 = df

df = pd.read_csv('c:/data/sahkonkulutus_2011.csv')
df = df.rename(columns={'Lopetusaika UTC':'date','Sähkön kulutus Suomessa':'Electricity Consumption'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
df = df.set_index('date')
df = df[['Electricity Consumption']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_elec_2011 = df

# Yhdistä 2010 datat train dataksi
#df_2010 = pd.concat(df_temps_2010, df_elec_2010, how='right', on=['date'])
#df_2010 = pd.concat([df_temps_2010, df_elec_2010], ignore_index=True)
#df_2010 = pd.concat([df_temps_2010, df_elec_2010], join='inner')
#df_2010 = pd.concat([df_temps_2010, df_elec_2010], axis=1)
df_2010 = pd.concat([df_temps_2010, df_elec_2010], axis=1, ignore_index=True)
#df_2010 = df_temps_2010.append(df_elec_2010, ignore_index=False)
print(df_2010.head(10))


#df_2010 = df_temps_2010.append(df_elec_2010, ignore_index=False)
print(df_2010.head(10))


df_2020 = df_temps_2010
df_2020['Electricity Consumption'] = df_elec_2010[['Electricity Consumption']]
print(df_2020.head())
# Yhdistä 2011 datat test dataksi


#%%
# Tehtävän vastaukset.  Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. 
# Luo tarvittavat osat viivakaavioon fig - muuttujaan matplotlib - kirjastoa käyttäen. Tuloksena pitäisi olla viivakaavio, jossa on piirrettynä oikeat arvot ja neuroverkon ennustukset.
fig