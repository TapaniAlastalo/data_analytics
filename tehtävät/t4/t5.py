import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/epl4.txt', sep = ',', decimal='.')

print(df)

df2 = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/sijat.txt', sep = ',', decimal='.')

print(df2)