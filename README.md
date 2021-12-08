# AgroX # Fertlizer Prediction 
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
data=pd.read_csv("../input/fertilizer-prediction/Fertilizer Prediction.csv")
data 

data.describe(include = 'all')

y=data['Fertilizer Name']
X=data.drop('Fertilizer Name',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,shuffle=True,random_state=1)
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
onehotencoding=ColumnTransformer(transformers= [
            ("onehotencoding", OneHotEncoder(sparse=False),["Soil Type","Crop Type"]) ],remainder="passthrough")


model = Pipeline(steps=[
              ("onehotencoding",onehotencoding),
              ("scaler", StandardScaler()),
              ("classifier", RandomForestClassifier())  
])
model.fit(X_train,y_train)
print("Model Accuracy : {:.2f}%".format(model.score(X_test,y_test)*100))
y_pred=model.predict(X_test)

clr=classification_report(y_test,y_pred)
print(clr)

