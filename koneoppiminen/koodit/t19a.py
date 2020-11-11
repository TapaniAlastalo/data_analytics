import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/MachineData.csv', sep=';', decimal='.', encoding='utf-8')
 
teamId = {'TeamA':1,  'TeamB':2, 'TeamC':3}
df['TeamId'] = df['Team'].map(teamId)
df['TeamId'].fillna(-1, inplace=True)

providerId = {'Provider1':1, 'Provider2':2, 'Provider3':3, 'Provider4':4}
df['ProviderId'] = df['Provider'].map(providerId)
df['ProviderId'].fillna(-1, inplace=True)

#print(df['Pclass'].unique())
#for col in df:
 #   print(col)
  #  print(df[col].unique())
    
df_train = df.sample(n = 200, replace = False) 
df_test = df.drop(df_train.index)

input_variables = ['TeamId', 'ProviderId', 'Lifetime', 'PressureInd', 'MoistureInd', 'TemperatureInd']

# LOGISTIC REGRESSION
# train
X = np.array(df_train[input_variables])
y = np.array(df_train['Broken'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_scaled, y)

# test
X = np.array(df_test[input_variables])
y = np.array(df_test['Broken'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['LRennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
# train
X = np.array(df_train[input_variables])
y = np.array(df_train['Broken'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC()
model.fit(X_scaled, y)

# test
X = np.array(df_test[input_variables])
y = np.array(df_test['Broken'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['SVMennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
# train
X = np.array(df_train[input_variables])
y = np.array(df_train['Broken'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier()
model.fit(X_scaled, y)

# test
X = np.array(df_test[input_variables])
y = np.array(df_test['Broken'])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['KNNennuste'] = ennuste

results_fields = ['Machine ID', 'Broken', 'LRennuste', 'SVMennuste', 'KNNennuste']
df_results = df_test[results_fields].sample(20)
