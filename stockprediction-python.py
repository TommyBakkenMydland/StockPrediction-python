# Install dependencies
# https://www.python.org/downloads/
# https://www.anaconda.com/distribution/#download-section
# How To Run Jupyter Notbook https://www.codecademy.com/articles/how-to-use-jupyter-notebooks#:~:targetText=To%20create%20a%20new%20notebook,running%20ones%20will%20be%20grey.


# pip install cython
# pip install -U scikit-learn
# pip install quandl

import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# Get the stock data
df = quandl.get("WIKI/FB")
# Print the data
print(df.head())

# Get the Adjusted close Price
df = df[['Adj. Close']]
print(df.head()) 

#A variable for predicting 'n' days out in the future
forecast_out = 30
# Create another colum (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
print(df.head())
print(df.tail())

### Create the independent data set (x) ###
# Convert the dataframe to numpy array
x = np.array(df.drop(['Prediction'], 1))
# Remove the last 'n' rows 
x = x[:-forecast_out]
print(x)

### Create the dependent data set (y) ###
# Convert the dataframe t a numpy array (All of the values including the NaN's)
y = np.array(df['Prediction'])
# Get the all of the y values except the last 'n' rows
y = y[:-forecast_out]
print(y)


### Split the data into 80% training and 20 % testing ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

### Create and train the Support Vector Machine (Regressor) ###
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

### Testing Model: Score return the coefficient of determination of R^2 of the prediction.
# The best possible score 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

### Create and train the Linear Regression Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

### Testing Model: Score return the coefficient of determination of R^2 of the prediction.
# The best possible score 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

### Set x_forecast equal to the last 30 rows of the original data set from Adj. Close Column
x_forcast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
print(x_forcast)

### Print linear regression model the prediction for the 'n' days
lr_prediction = lr.predict(x_forcast)
print(lr_prediction)

### Print support vector regressor model the prediction for the 'n' days
svm_prediction = svr_rbf.predict(x_forcast)
print(svm_prediction)