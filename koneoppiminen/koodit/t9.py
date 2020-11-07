import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/Titanic.csv', sep=',', encoding='utf-8')
 
sex_B = {'male':0,  'female':1}
df['Sex_B'] = df['Sex'].map(sex_B)

df['Age'].fillna(-1, inplace=True)

embarked_B = {'C':0,  'S':1, 'Q':2}
df['Embarked_B'] = df['Embarked'].map(embarked_B)
df['Embarked_B'].fillna(-1, inplace=True)

#print(df['Pclass'].unique())
#for col in df:
 #   print(col)
  #  print(df[col].unique())
    
df_train = df.sample(n = 200, replace = False) 
df_test = df.drop(df_train.index)

input_variables = ['Pclass', 'Sex_B', 'Age', 'SibSp', 'Parch', 'Embarked_B']

# LOGISTIC REGRESSION
# train
X = np.array(df_train[input_variables])
y = np.array(df_train['Survived'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_scaled, y)

# test
X = np.array(df_test[input_variables])
y = np.array(df_test['Survived'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['LRennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
# train
X = np.array(df_train[input_variables])
y = np.array(df_train['Survived'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC()
model.fit(X_scaled, y)

# test
X = np.array(df_test[input_variables])
y = np.array(df_test['Survived'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['SVMennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
# train
X = np.array(df_train[input_variables])
y = np.array(df_train['Survived'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier()
model.fit(X_scaled, y)

# test
X = np.array(df_test[input_variables])
y = np.array(df_test['Survived'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['KNNennuste'] = ennuste

results_fields = ['PassengerId', 'Survived', 'LRennuste', 'SVMennuste', 'KNNennuste']
df_results = df_test[results_fields].sample(20)
