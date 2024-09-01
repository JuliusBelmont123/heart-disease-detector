import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#filter warnings
import warnings
warnings.filterwarnings("ignore",message="X does not have valid feature names, but LogisticRegression was fitted with feature names")
#csv to pandas
heart_data=pd.read_csv('heart.csv')
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=3)
model=LogisticRegression(max_iter=1000)
#training
model.fit(X_train,Y_train)
#store
with open('model.pkl','wb') as f:
   pickle.dump(model,f)


#accuracy on training data
#X_train_pred=model.predict(X_train)
#train_data_acc=accuracy_score(X_train_pred,Y_train)

#accuracy on test data
#X_test_pred=model.predict(X_test)
#test_data_acc=accuracy_score(X_test_pred,Y_test)

#input_data = (58,1,0,114,318,0,2,140,0,4.4,0,3,1)
#change input data to numpy array
#input_data_as_numpy_array=np.asarray(input_data)
#reshape numpy array as prediction for one instance
#input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#prediction=model.predict(input_data_reshaped)
#if (prediction[0]==0):
#   print("No ",end='')
#print("Heart Disease Detected")
