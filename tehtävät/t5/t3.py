import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta5/golf.zip', sep = ',', decimal='.')
# korvataan puuttuvat arvot nollilla
df.fillna(0, inplace=True)
#print(df.iloc[:,30:40])