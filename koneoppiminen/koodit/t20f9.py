import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/Telco.csv', sep=';', decimal='.', encoding='utf-8')
df.fillna(0, inplace=True)

#input_variables = ['region', 'tenure', 'age', 'marital', 'income', 'employ', 'gender', 'tollfree', 'wireless', 'cardten', 'logtoll', 'logcard', 'custcat']
input_variables = df.iloc[:,1:40]
predict_field = "churn"
 
df_train = df.sample(n = 900, replace = False) 
df_test = df.drop(df_train.index)

# train
X = np.array(df_train.iloc[:,1:40])
y = np.array(df_train[predict_field])

# LOGISTIC REGRESSION
# train
X = np.array(df_train.iloc[:,1:40])
y = np.array(df_train[predict_field])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_scaled, y)

# test
X = np.array(df_test.iloc[:,1:40])
y = np.array(df_test[predict_field])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['LRennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
# train
X = np.array(df_train.iloc[:,1:40])
y = np.array(df_train[predict_field])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC()
model.fit(X_scaled, y)

# test
X = np.array(df_test.iloc[:,1:40])
y = np.array(df_test[predict_field])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['SVMennuste'] = ennuste


# SUPPORT VECTOR CLASSIFIER
# train
X = np.array(df_train.iloc[:,1:40])
y = np.array(df_train[predict_field])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier()
model.fit(X_scaled, y)

# test
X = np.array(df_test.iloc[:,1:40])
y = np.array(df_test[predict_field])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

ennuste = model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df_test['KNNennuste'] = ennuste

results_fields = ['region', 'tenure', 'age', 'marital', 'income', 'employ', 'gender', 'tollfree', 'wireless', 'cardten', 'logtoll', 'logcard', 'custcat','churn', 'LRennuste', 'SVMennuste', 'KNNennuste']
df_results = df_test[results_fields].sample(20)