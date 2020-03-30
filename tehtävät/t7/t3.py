import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta7/teht5.txt', sep=",", decimal='.')

#print("Puuttuvien arvojen lukumäärä per muuttuja (1):\n%s" % df.isnull().sum())
df['hissi']=df['hissi'].map({'ei': 0, 'on': 1})
df['kunto']=df['kunto'].map({'huono': 0, 'tyyd.': 1, 'hyvä': 0})
df['keskusta'] = 0
df.loc[(df['kaupunginosa']=='Keskusta'), 'keskusta'] = 1
#print("Puuttuvien arvojen lukumäärä per muuttuja (2):\n%s" % df.isnull().sum())

median_kunto = df["kunto"].median()
df["kunto"].fillna(median_kunto, inplace=True)
print("Puuttuvien arvojen lukumäärä per muuttuja (3):\n%s" % df.isnull().sum())
#print(df)

x = df[['m2','rakennusvuosi','hissi','kunto','keskusta']]
y = df['hinta']

# jaotellaan testi- / opetusdata
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.4, random_state = 31)
print(xTrain.shape)
print(xTest.shape)

# luodaan malli-olio opetusdatalla
model = LinearRegression()
model.fit(xTrain,yTrain)

# Testaus testidatalla
print("Testausdatan selityskerroin:", model.score(xTest,yTest))

forecast = model.predict(xTest)
plt.scatter(yTest, forecast)
plt.xlabel('Havaittu')
plt.ylabel('Ennustettu')
plt.show()