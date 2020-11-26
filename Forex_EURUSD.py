
******** IF You need this script, pls be in touch *********
***********************************************************


#Import the libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
os.environ["ALPHAVANTAGE_API_KEY"] = "XXXXXXXXXXXXXXX"
plt.style.use('fivethirtyeight')

from datetime import datetime
date=str(datetime.now().strftime('%d-%m-%Y'))
time = str (datetime.now().strftime('%H:%M:%S'))
year = int(datetime.now().strftime('%Y'))
month = int(datetime.now().strftime('%m'))
day = int(datetime.now().strftime('%d'))

#Get the stock quote
from alpha_vantage.foreignexchange import ForeignExchange
import matplotlib.pyplot as plt
cur1 = 'EUR'
cur2 = 'USD'
fx = ForeignExchange(key="ALPHAVANTAGE_API_KEY", output_format='pandas')
df, meta_data = fx.get_currency_exchange_intraday(cur1, cur2, '60min', 'full')
df=df.iloc[::-1]


#Create new dataframe with only the 'Close column'
data = df.filter(['4. close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len  = math.ceil(len(dataset)* .8)
training_data_len

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len,:]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])

#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data, as LSTM needs 3 Dimensional, need to reshape..
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
##x_train.shape




#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


#Convert the data to a numpy array so that can be used in the LSTM model
x_test = np.array(x_test)

#need to change to 3D shape as expected to LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test )**2 )


#Get the quote (THIS IS MODEL PREDICTS - Dont change start datetime)
##stock_quote = web.DataReader(pair, "av-forex-daily", start=datetime(2015, 2, 9), end=datetime(year, month, day), api_key=os.getenv('XXXXXXXXXXXXX'))
stock_quote, meta_data = fx.get_currency_exchange_intraday(cur1, cur2, '60min', 'full')
stock_quote=stock_quote.iloc[::-1]
#Create a new dataframe
new_df=stock_quote.filter(['4. close'])
#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = str(scaler.inverse_transform(pred_price))
pred_price = pred_price[2:-2]
print ("ML Predicted Price: " , pred_price , "\n" )

#**Get the quote (REAL_WORLD / ACTUAL PRICE)
stock_quote2, meta_data = fx.get_currency_exchange_intraday(cur1, cur2, '60min', 'full')
stock_quote2 = stock_quote.iloc[::-1]
#Create a new dataframe
new_df=stock_quote2.filter(['4. close'])
#get the value fr the df
actual_price = new_df['4. close'].values[0]

## Working
from openpyxl import load_workbook

new_row_data = [
    [date, time, pred_price, actual_price, rmse ]] 

wb = load_workbook("C:/Users/XXXXX/OneDrive/AzureMLExcel/ForexStockMonitoring.xlsx")
# Select  Worksheet no.
ws = wb.worksheets[3]

# Append 2 new Rows - Columns A - D
for row_data in new_row_data:
    # Append Row Values
    ws.append(row_data)

wb.save("C:/Users/XXXXXX/OneDrive/AzureMLExcel/ForexStockMonitoring.xlsx")
#wb.save("MonitorAccuracy.xlsx")
wb.close()
