import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../../data/diabetes.csv')
# poista NaN rivit
df.dropna(inplace=True)

# poista duplikaatit
df.drop_duplicates(keep='first', inplace=True)

# korvaa suuresti poikkeavat arvot
# laske mediaani ja korvaa 0 arvot sillä ja överi poikkeamat (max 2x median).
replaceableValues = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for value in replaceableValues:    
    median = df[value][df[value] != 0].median()
    maxM = 2.0 * median
    df.loc[(df[value] == 0), value]= median
    df.loc[(df[value] > maxM), value]= maxM
    #dfplot = df[value]
    #dfplot.plot.bar(figsize=(10, 4))
    #plt.show()


# sekoita pakka
from sklearn.utils import shuffle
df = shuffle(df)

# muuta data valvotun oppimisen muotoon, eli erottele data ominaisuuksiin ja ennustettaviin arvoihin.
# ominaisuudet sarakkeet 0-6; Sarake 7 ikä jää pois, koska nuoret eivät ole vielä kerenneet saada diabetestä, joten ikä sotkisi ennusteen. Lisäksi osa vanhoista ei kerkeä saamaan diabetestä ennen oletettua kuolemaa.
x = np.array(df.iloc[:, 0:7])
# ennustettava arvo sarake 8
y = np.array(df.iloc[:, 8])

# skaalaa standardoimalla / minimi-maksimi menetelmällä
scaler = StandardScaler()
#scaler = MinMaxScaler()
scaledX = scaler.fit_transform(x.reshape(-1, 7))

# Jaa datasetti koulutus- ja testidatasettiin, käytetään skaalattua x arvoa SEKÄ opetus-, että testidatassa!
limiter = 400
train_x = scaledX[:limiter]
test_x = scaledX[limiter:]

train_y = y[:limiter]
test_y = y[limiter:]


# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. Sijoita muokkaamasi dataframe df - muuttujaan.
print("Amount of NaN rows in dataframe: ",df.isna().any().sum())
print("Amount of duplicate rows in dataframe: ",df.duplicated().any().sum())