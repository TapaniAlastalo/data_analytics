import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data/Mall_Customers.csv')

fields = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = np.array(df[fields])

model = KMeans(n_clusters=4)
model.fit(X)

labels = model.labels_
df['Label'] = labels

colors = {0:'red', 1:'blue', 2:'green', 3:'magenta'}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0,4):
    x = df.loc[df['Label'] == i][fields[0]].values
    y = df.loc[df['Label'] == i][fields[1]].values
    z = df.loc[df['Label'] == i][fields[2]].values
    ax.scatter(x, y, z, marker='o', s=40, color=colors[i], label='Customer class '+str(i+1))

ax.set_xlabel(fields[0])
ax.set_ylabel(fields[1])
ax.set_zlabel(fields[2])
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2))
plt.show()