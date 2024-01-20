from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

X = iris.drop('target',axis=1)
y = iris['target']

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=25)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
model = knn.fit(X_train,Y_train)

y_pred = model.predict(X_test)
print("Predictions : ",y_pred)
accuracy= accuracy_score(Y_test,y_pred)
print("Accuracy : ", accuracy)

joblib.dump(model,"./model.pkl")
