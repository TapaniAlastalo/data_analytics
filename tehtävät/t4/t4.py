import pandas as pd

#df = pd.read_csv('http://gpspekka.kapsi.fi/trafi56.zip', sep = ',', decimal='.')
df = pd.read_csv('./testi/trafi56.zip', sep = ';', decimal=',', encoding='latin_1', nrows=10000)

print(df)