import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/kysely.csv', sep = ',', decimal='.')

print(df)