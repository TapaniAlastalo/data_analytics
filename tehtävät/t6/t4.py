import pandas as pd
import datetime

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta6/km.txt', sep=",", decimal='.')
print(df)
