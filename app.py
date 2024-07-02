import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from knn import Knn
df = pd.read_csv("Social_Network_Ads.csv")

df.drop(columns=["User ID"],inplace=True)
le = LabelEncoder()
df['Gender'] =le.fit_transform(df['Gender'])
ss = StandardScaler()
X = df.iloc[:,0:df.shape[1]-1]
y = df.iloc[:,-1]
X = ss.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
# print(X_train.shape,X_test.shape)
# this is calculating accuracy score from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
ypred = knn.predict(X_test)
print("Accuracy Score of library: ",accuracy_score(y_test,ypred))
# from my class code
apnaknn = Knn(k=5)
apnaknn.fit(X_train,y_train)
y_pred1 = apnaknn.predict(X_test)
print("Accuracy Score of Scratch: ",accuracy_score(y_test,y_pred1))

