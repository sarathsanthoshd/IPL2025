import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor   
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
import pickle

df = pd.read_csv("D:\IDS\ipl_score_prediction\ipl.csv")
df.head()

print("Shape:",df.shape)
print("Dtype:",df.dtypes)
print("DAta set coloumns:",df.columns)

df.isna().sum()

df.duplicated().sum()
#including powerplayy 6 overs!!
print("Before removing first 6 overs:",df.shape)
#Excluding powerplay 6 overs!!
df_without_six = df[df['overs'] >= 6.0]
print("After removing first 6 overs:",df_without_six.shape)

#here Y is tgt variable-total is the reqired tgt
Y_without_six = df_without_six['total']
print("Target variable 'Y':",Y_without_six)
#X is remaining colmns..!!
X_without_six = df_without_six.drop(columns=['total','mid','venue','batsman','bowler','striker','non-striker','date'])
print(X_without_six.columns)
print(X_without_six.shape)

encoded_X_without_six =pd.get_dummies(X_without_six)

print("Encoded coloumns:",encoded_X_without_six.head())

print("Encoded Dset Shape:",encoded_X_without_six.shape)





train_x,test_x,train_y,test_y=train_test_split(encoded_X_without_six,Y_without_six,test_size=0.2)
print(f"X_train:{train_x.shape}")
print(f"X_test:{test_x.shape}")
print(f"Y_train:{train_y.shape}")
print(f"Y_test:{test_y.shape}")

model_dt =DecisionTreeRegressor()
model_dt.fit(train_x,train_y)

test_x_pred = model_dt.predict(test_x)

print("Mean Absolute Error (MAE):",mae(test_y,test_x_pred))
print("Mean Squared Error (MSE):",mse(test_y,test_x_pred))
print("Root Mean Squared Error (RMSE):",np.sqrt(mse(test_y,test_x_pred)))

pickle.dump( model_dt , open('model.pkl', 'wb'))

#XG-Boost(optional testing--purpose)

model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model_xgb.fit(train_x, train_y)

test_x_pred = model_xgb.predict(test_x)

print("Mean Absolute Error in XGB(MAE):", mae(test_y, test_x_pred))
print("Mean Squared Error in XGB(MSE):", mse(test_y, test_x_pred))
print("Root Mean Squared Error in XGB (RMSE):", np.sqrt(mse(test_y, test_x_pred)))

## here XGB was not inerted/load to pickle in p.flask