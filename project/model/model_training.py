import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("heart.csv")

print(df)

x = df.drop("target", axis=1)
y = df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)

x_test_scaler = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train_scaler,y_train)
y_pred = model.predict(x_test_scaler)

print(accuracy_score(y_test, y_pred))

with open('heart_disease_model.pkl',"wb") as model_file:
    pickle.dump(model,model_file)

with open("scaler.pkl",'wb') as scaler_file:
    pickle.dump(scaler,scaler_file)

print("model and scaler saved successfully")
