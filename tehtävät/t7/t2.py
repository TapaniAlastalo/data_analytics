import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta7/teht2.txt', sep=",", decimal='.')
df['sukupuoli']=df['sukupuoli'].map({'nainen': 0, 'mies': 1})
print(df)

x = df[['tulot','naimisissa','sukupuoli']]
y = df['ostaa']

# luodaan malli-olio
#model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model = DecisionTreeClassifier(max_depth=3)
# sovitetetaan, eli generoidaan päätöspuu
model.fit(x,y)

# ennustetaan tulokset
y_pred = model.predict(x)
# katsotaan tarkkuus
print("Accuracy:", model.score(x,y))
# Confusion Matrix
print(confusion_matrix(y, y_pred))


export_graphviz(decision_tree=model, out_file="tree_ostaako.dot",
                feature_names=x.columns, class_names=True, filled=True, rounded=True)