import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

days_to_forecast = 7

df = pd.read_csv('data/Google_Stock_Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = df.apply(lambda row: len(df) - row.name, axis=1)
df['CloseFuture'] = df['Close'].shift(days_to_forecast)


df_test = df[:185]
df_train = df[185:]

X = np.array(df_train[['Time', 'Close']])
#X = X.reshape(-1,1) # vain jos yksiulotteinen taulukko
y = np.array(df_train['CloseFuture'])


model = linear_model.LinearRegression()
model.fit(X, y)
ennuste_train = model.predict(X)
df_train['Ennuste'] = ennuste_train


X_test = np.array(df_test[['Time', 'Close']])
#X_test = X_test.reshape(-1,1) # vain jos yksiulotteinen taulukko
ennuste_test = model.predict(X_test)
df_test['Ennuste'] = ennuste_test


plt.scatter(df['Date'].values, df['Close'].values, color='black', s=1)
plt.plot((df_train['Date'] + pd.DateOffset(days=days_to_forecast)).values, df_train['Ennuste'].values, color='blue', linewidth=1)
plt.plot((df_test['Date'] + pd.DateOffset(days=days_to_forecast)).values, df_test['Ennuste'].values, color='red', linewidth=1)

plt.show()

df_train_validation = df_train.dropna()
df_test_validation = df_test.dropna()

print("Ennusteen keskivirhe opetusdatassa on %.f" % mean_absolute_error(df_train_validation['CloseFuture'], df_train_validation['Ennuste']))
print("Ennusteen keskivirhe testidatassa on %.f" % mean_absolute_error(df_test_validation['CloseFuture'], df_test_validation['Ennuste']))

print('Mallin kertoimet ovat \n', model.coef_, model.intercept_)

#print(type(df['Date'][0]))
# ennuste = 0.173 * time + 0.588 * close + 161  # 30 päivän ennusteella