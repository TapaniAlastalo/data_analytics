import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/fruit_data.csv', sep=',', encoding='utf-8')

X = np.array(df[['mass', 'width', 'height', 'color_score']])

fruit_codes = {'apple':0,  'lemon':1, 'mandarin':2, 'orange':3}
df['fruit_code'] = df['fruit_name'].map(fruit_codes)

y = np.array(df['fruit_code'])

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# LOGISTIC REGRESSION
model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_scaled, y)
ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df['LRennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
model = SVC()
model.fit(X_scaled, y)
ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df['SVMennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
model = KNeighborsClassifier()
model.fit(X_scaled, y)
ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df['KNNennuste'] = ennuste
