import pandas as pd

dfPyorat = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/pyorat.txt', sep = ',', decimal='.')
dfSaa = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/helsinki2017.csv', sep = ',', decimal='.')

print("Testi:")
#print(dfPyorat)
dfPyorat.rename(columns={'Month': 'Kk', 'Day': 'Pv', 'Hour':'Klo'}, inplace=True)
#dfPyorat['Klo'] = (dfPyorat['Klo']).astype(int)
print("&")
dfSaa['Klo'] = (dfSaa['Klo'].str.split(':').str.get(0)).astype(int)
#print(dfSaa)
print("---")

#dfYhd = pd.concat([dfPyorat, dfSaa], axis=1)
tulostettavat = ['Kk', 'Pv', 'Klo']
#print(dfYhd[tulostettavat])
print("combine")
dfYhd2 = dfPyorat.combine_first(dfSaa)
print(dfYhd2[tulostettavat])
print(dfYhd2)