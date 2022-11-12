import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import decomposition
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
import pickle

df_co = pd.read_csv('Transformed_Co', index_col = 0)
df_pt = pd.read_csv('Transformed_Pt' , index_col = 0)
df_co_len = df_co.shape[0]
df_pt_len = df_pt.shape[0]
df_co_pca = pd.DataFrame(df_co)
df_pt_pca = pd.DataFrame(df_pt)
y1 = pd.Series([0]*df_co_len)
y1.shape
y2 = pd.Series([1]*df_pt_len, index = range(df_co_len-1,(df_co_len + df_pt_len)-1))
y2.shape
y = pd.concat([y1,y2]) 
y.shape

X = pd.concat([df_co_pca, df_pt_pca])

X_train, X_test, y_train1, y_test1 = train_test_split(X, y, random_state = 41)
y_train = pd.DataFrame(y_train1)
y_test = pd.DataFrame(y_test1)

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)

X_train, X_test, y_train1, y_test1 = train_test_split(X, y)
y_train = pd.DataFrame(y_train1)
y_test = pd.DataFrame(y_test1)

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
    
knn = KNeighborsClassifier()
knn.fit(X_train, np.ravel(y_train))

grid_values = {'n_neighbors': [2]}
grid_clf = GridSearchCV(knn, param_grid=grid_values, scoring='roc_auc')
grid_clf.fit(X_train, y_train1)
prediction = grid_clf.predict(X_test)

prec = precision_score(y_test, prediction)
rec = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)

print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("F1: {}".format(f1))

y_pred= knn.predict(X_test)
print(classification_report(y_test,y_pred))
print(y_test[0].tolist())
print(y_pred.tolist())


pickle.dump(knn,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
