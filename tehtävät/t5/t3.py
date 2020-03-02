import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta5/lam_raw.csv', sep = ';', decimal='.')
df.columns= ['pistetunnus', 'vuosi', 'päivännro', 'tunti', 'minuutti', 'sekunti', 'sadasosa', 'pituus', 'kaista', 'suunta', 'ajoneuvoluokka', 'nopeus', 'fault', 'kokonaisaika', 'aikaväli', 'jonoalku']
df.drop(df[df['fault'] > 0].index, inplace=True)

#sns.pairplot(df[['suunta', 'ajoneuvoluokka', 'tunti', 'nopeus']].dropna(), kind='reg')

sns.pairplot(data = df, vars = ['suunta', 'tunti', 'nopeus'], hue = 'ajoneuvoluokka', diag_kind = 'kde', height = 4)

# Create an instance of the PairGrid class.
#grid = sns.PairGrid(data = df, vars = ['suunta', 'ajoneuvoluokka', 'tunti'], height = 4)

# Map a scatter plot to the upper triangle
#grid = grid.map_upper(plt.scatter, color = 'darkred')
# Map a histogram to the diagonal
#grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', edgecolor = 'k')
# Map a density plot to the lower triangle
#grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')

plt.show()