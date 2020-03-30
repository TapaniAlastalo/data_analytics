import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta7/nba_logreg.csv', sep=",", decimal='.')

#print(df)
#print("Puuttuvien arvojen lukumäärä per muuttuja (1):\n%s" % df.isnull().sum())
# korvataan puuttuvat 3P% mediaanilla
median_3P = df["3P%"].median()
df["3P%"].fillna(median_3P, inplace=True)
#print("Puuttuvien arvojen lukumäärä per muuttuja (2):\n%s" % df.isnull().sum())
#print(df)
#print(df.iloc[:, 1:21])

x = df.iloc[:, 1:20] # 0.716 / 0.713
#x = df.iloc[:, np.r_[2:20]] # 0.692
#x = df.iloc[:, 1:19] # 0.717
#x = df.iloc[:, np.r_[1:17, 19]] # 0.722
#x = df.iloc[:, np.r_[1,2,3,5,6,7,8,9,10,11,13,16,17,19]] # 0.722
y = df['TARGET_5Yrs']

scaler = StandardScaler()
xScaled = scaler.fit_transform(x)

# luodaan logistinen regreassio malli
model = LogisticRegression()
model.fit(xScaled,y)

# ennustetaan tulokset
y_pred = model.predict(xScaled)
# katsotaan tarkkuus
print("Accuracy:", model.score(xScaled,y))
# Confusion Matrix
print(confusion_matrix(y, y_pred))


# luodaan päätöspuu malli
model2 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
# sovitetetaan, eli generoidaan päätöspuu
model2.fit(xScaled,y)

# ennustetaan tulokset
y_pred2 = model2.predict(xScaled)
# katsotaan tarkkuus
print("Accuracy 2:", model2.score(xScaled,y))
# Confusion Matrix
print(confusion_matrix(y, y_pred2))

#print(model.predict_proba(xScaled))

# kenttien vaikutus dataan
print(df.corrwith(y))