import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta7/teht1.txt', sep=",", decimal='.')

df['s'] = 0
df.loc[(df['sauna']!='ei'), 's'] = 1
df['sauna']=df['s']
del df['s']
#print(df)

x = df[['ala','makuuhuoneita','sauna']]
y = df['hinta']

model = LinearRegression()
model.fit(x,y)
print("Selityskerroin: ", model.score(x,y))

x2 = df[['ala','makuuhuoneita']]
model.fit(x2,y)
print("Selityskerroin2: ", model.score(x2,y))
x3 = df[['ala','sauna']]
model.fit(x3,y)
print("Selityskerroin3: ", model.score(x3,y))
x4 = df[['makuuhuoneita','sauna']]
model.fit(x4,y)
print("Selityskerroin4: ", model.score(x4,y))

#---- PARAS----
model.fit(x,y)

forecast = model.predict(x)
plt.scatter(y, forecast)
plt.xlabel('Havaittu')
plt.ylabel('Ennustettu')
plt.show()

