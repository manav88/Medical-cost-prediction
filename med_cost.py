import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
df=pd.read_csv("insurance.csv")


tem=pd.get_dummies(df["region"])

df.drop("region",axis=1,inplace=True)
df=pd.concat([df,tem],axis=1)
print(df.head(10))

map={"yes":1,"no":0}
df["smoker"]=df["smoker"].map(map)
map1={"female":0,"male":1}
df["sex"]=df["sex"].map(map1)
print(df.head(10))
df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=2)
plt.show()

x=df["smoker"]
y=df["expenses"]
plt.figure(figsize=(12,9))
plt.scatter(x,y)
plt.xlabel("Non Smoker Vs Smoker")
plt.ylabel("Charges")

Y=df["charges"]
X=df.drop("charges",axis=1)


from sklearn.model_selection import train_test_split
#Splitting the data into 85% for training and 15% for testing
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.15)
from sklearn.linear_model import LinearRegression
#Training a multiple linear regression model
reg=LinearRegression().fit(x_train,y_train)
y_pred=reg.predict(x_test)



from sklearn.metrics import r2_score
#Checking the R squared error on test data
r2_score(y_test,y_pred)



# Storing independent features in a temporary variable
P_X=X

from sklearn.preprocessing import PolynomialFeatures
#Changing the data to a 3rd degree polynomial
pol=PolynomialFeatures(degree=3)
P_X=pol.fit_transform(X)
P_X
#Training the model similarly but with 3rd degree polynomial of X this time
x_train,x_test,y_train,y_test=train_test_split(P_X,Y,random_state=1,test_size=0.15)
reg=LinearRegression().fit(x_train,y_train)
y_pred=reg.predict(x_test)
r2_score(y_test,y_pred)


#Cross validating the score to check and avoid overfitting
from sklearn.model_selection import cross_val_score
c=cross_val_score(reg,P_X,Y,cv=4)
c

# Final Mean Accuracy
print("Mean accuracy after cross validation is:",c.mean()*100,end="%")
