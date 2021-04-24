# In[1]:
#importing the dataset
import pandas as pd
data = pd.read_csv('SolarPrediction.csv')
data = data.sort_values(['UNIXTime'], ascending = [True])

#del data['avg_zenith_angle']
#del data['UNIXTime']
#del data['Time']
#del data['TimeSunRise']
#del data['TimeSunSet']
from datetime import datetime
from pytz import timezone
import pytz
hawaii= timezone('Pacific/Honolulu')
data.index =  pd.to_datetime(data['UNIXTime'], unit='s')
data.index = data.index.tz_localize(pytz.utc).tz_convert(hawaii)
data['MonthOfYear'] = data.index.strftime('%m').astype(int)
data['DayOfYear'] = data.index.strftime('%j').astype(int)
data['WeekOfYear'] = data.index.strftime('%U').astype(int)
data['TimeOfDay(h)'] = data.index.hour
data['TimeOfDay(m)'] = data.index.hour*60 + data.index.minute
data['TimeOfDay(s)'] = data.index.hour*60*60 + data.index.minute*60 + data.index.second
data['TimeSunRise'] = pd.to_datetime(data['TimeSunRise'], format='%H:%M:%S')
data['TimeSunSet'] = pd.to_datetime(data['TimeSunSet'], format='%H:%M:%S')
data['DayLength(s)'] = data['TimeSunSet'].dt.hour*60*60 \
                           + data['TimeSunSet'].dt.minute*60 \
                           + data['TimeSunSet'].dt.second \
                           - data['TimeSunRise'].dt.hour*60*60 \
                           - data['TimeSunRise'].dt.minute*60 \
                           - data['TimeSunRise'].dt.second
data.drop(['Data','Time','TimeSunRise','TimeSunSet'], inplace=True, axis=1)
data.head()

# In[2]:
#Separating the Independent and Dependent Variables
X = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayOfYear', 'TimeOfDay(s)']]
y = data['Radiation']

# In[3]:
#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# In[4]:
#Fitting the Multiple Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

#Predcution based on the test in MLR
predicted = model.predict(X_test)
print(predicted)

# In[5]:
#Predicting the Test Set for MLR
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np
r_squared_mlr = r2_score(Y_test, predicted)
variance_score_mlr = explained_variance_score(Y_test, predicted)
print("MAE:"+str(mean_absolute_error(Y_test, predicted)))
print("MSE:"+str(mean_squared_error(Y_test, predicted)))
print('variance score for MLR = {}'.format(variance_score_mlr))
print("RMSE:"+str(np.sqrt(mean_squared_error(Y_test, predicted))))
print('r2 score for MLR = {}'.format(r_squared_mlr))

# In[6]:
#Fitting the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
rf_reg = RandomForestRegressor(n_estimators = 100)
rf_reg.fit(X_train, Y_train)



feature_importances = rf_reg.feature_importances_

X_train_opt = X_train.copy()
removed_columns = pd.DataFrame()
models = []
r2s_opt = []

for i in range(0,5):
    least_important = np.argmin(feature_importances)
    removed_columns = removed_columns.append(X_train_opt.pop(X_train_opt.columns[least_important]))
    rf_reg.fit(X_train_opt, Y_train)
    feature_importances = rf_reg.feature_importances_
    accuracies = cross_val_score(estimator = rf_reg,
                                 X = X_train_opt,
                                 y = Y_train, cv = 5,
                                 scoring = 'r2')
    r2s_opt = np.append(r2s_opt, accuracies.mean())
    models = np.append(models, ", ".join(list(X_train_opt)))
    
feature_selection = pd.DataFrame({'Features':models,'r2 Score':r2s_opt})
feature_selection.head()



# In[7]:
X_train_best = X_train[['Temperature', 'DayOfYear', 'TimeOfDay(s)']]
X_test_best = X_test[['Temperature', 'DayOfYear', 'TimeOfDay(s)']]
rf_reg.fit(X_train_best, Y_train)
#Predcution based on the test in RFR
randomforest_pred= rf_reg.predict(X_test_best)


# In[8]:
#Predicting the Test Set for RFR
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
variance_score_rfr = explained_variance_score(Y_test, randomforest_pred)
r_squared = r2_score(Y_test, randomforest_pred)
print("MAE:"+str(mean_absolute_error(Y_test, randomforest_pred)))
print("MSE:"+str(mean_squared_error(Y_test, randomforest_pred)))
print("RMSE:"+str(np.sqrt(mean_squared_error(Y_test, randomforest_pred))))
print('variance score for RFR = {}'.format(variance_score_rfr))
print('r2 score for RFR = {}'.format(r_squared))

# In[9]:
#Dump model regressior on the disk
from joblib import dump
dump(rf_reg,'forest.joblib')
dump(model,'linear.joblib')

# In[10]:

