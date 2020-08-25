import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta7/mushrooms.csv', sep=",", decimal='.')

#print(df)
#print("Puuttuvien arvojen lukumäärä per muuttuja (1):\n%s" % df.isnull().sum())
df2 = df.apply(LabelEncoder().fit_transform)
#print(df2)
#print(df2.iloc[:, 1:23])
x = df2.iloc[:, 1:23]
y = df['class']

# Ei tarvita - Skaalataan, jotta kaikki yhtä painavia
#scaler = StandardScaler()
#x = scaler.fit_transform(x)

# jaotellaan testi- / opetusdata
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.4, random_state = 31)
#print(xTrain.shape)
#print(xTest.shape)


# luodaan päätöspuu malli
model2 = DecisionTreeClassifier(criterion='entropy', max_depth=2)
model3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model4 = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)


# sovitetetaan, eli generoidaan päätöspuu
model2.fit(xTrain,yTrain)
model3.fit(xTrain,yTrain)
model4.fit(xTrain,yTrain)
model5.fit(xTrain,yTrain)


print("\nmax depth 2:")
print("Selityskerroin:", model2.score(xTest,yTest))
print(confusion_matrix(y, model2.predict(x)))

print("\nmax depth 3:")
print("Selityskerroin:", model3.score(xTest,yTest))
print(confusion_matrix(y, model3.predict(x)))

print("\nmax depth 4:")
print("Selityskerroin:", model4.score(xTest,yTest))
print(confusion_matrix(y, model4.predict(x)))

print("\nmax depth 5:")
print("Selityskerroin:", model5.score(xTest,yTest))
print(confusion_matrix(y, model5.predict(x)))
