import pandas as pd

dfPyorat = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/pyorat.txt', sep = ';', decimal=',')
dfSaa = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/helsinki2017.csv', sep = ';', decimal=',')

print("Testi:")
print(dfPyorat)
print("&")
print(dfSaa)