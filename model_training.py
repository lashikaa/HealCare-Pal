import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import joblib


df=pd.read_csv('pre_processed.csv')


df.columns= df.columns.str.strip()


X=df.drop(['Disease'], axis=1)
y=df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
svc_algo=SVC()
svc_algo.fit(X_train,y_train)


pred=svc_algo.predict(X_test)


joblib.dump(svc_algo,"svc_algorithm.joblib")