import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
df=pd.read_excel('bank_data.xlsx')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(max_depth=49, n_estimators=9000, learning_rate=0.01,n_jobs=-1)
xgb_clf.fit(x_train, y_train)
pred=xgb_clf.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(pred,y_test))
print(confusion_matrix(pred,y_test))
import pickle

pickle.dump(xgb_clf, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))
print(model.predict([[40,5.6,70,0,0,0,0,0,0,0,0]]))









