# ForexMonitoring

End-to-end Forex/stock prediction (personal project): 

Utilize Keras LSTM (Long Short Term memory) algorithm to forecast the closing price of major currency pairs every hourly.

Automate python scripts which hourly (on weekdays only) retrieve forex intraday panda dataset using API, train and test the time-series prediction model. 
The automation process is setup at a VPS (Virtual Private server) which is updated every hourly into a excel spreadsheet.

Finally perform data analysis of the forecast vs actual models through powerBI DAX, charts and visualization. 
Charts to derive insights to determine which time/day-range (morning, late morning, afternoon, night) has the most accuracy. 

Also, there is a daily monitoring stack chart to visualize the breakdown of the prediction accuracy based on total number of correct, wrong and no change.  Of course, there is a slicer to adjust how we wanted to view the breakdown - daily, monthly or yearly. 

Metric measurement is using:
i) RMSE (not shown in powerBI)
ii) Accuracy percentage (shown in powerBI)
iii) Accuracy percentage based on time range (shown in powerBI). 

Git repository URL: https://github.com/emilysmtan/ForexMonitoring

With reference from: 
https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html

