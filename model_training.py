# Import Libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import joblib

# Upload Dataset
df=pd.read_csv('pre_processed.csv')

# Striping the space in the column heading
df.columns= df.columns.str.strip()

# Splitting Input and Output variables
X=df.drop(['Disease'], axis=1)
y=df.iloc[:,-1]

# Splitting the Dataset into Testing and Training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Training the SVM algorithm
svc_algo=SVC()
svc_algo.fit(X_train,y_train)

# Predicitng the Dataset
pred=svc_algo.predict(X_test)

# Saving the algorithm trained for the use of Integration
joblib.dump(svc_algo,"svc_algorithm.joblib")



