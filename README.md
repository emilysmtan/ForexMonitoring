# ForexMonitoring

End-to-end Forex/stock prediction (personal project): 

Utilize Keras LSTM (Long Short Term memory) algorithm to forecast the closing price of major currency pairs every hourly.

Automate python scripts which hourly (on weekdays only) retrieve forex intraday panda dataset using API, train and test the time-series prediction model. 
The automation process is setup at a VPS (Virtual Private server) which is updated every hourly the dataset into a excel spreadsheet via a task scheduler.

Finally perform data analysis of the forecast vs actual models through powerBI DAX, charts and visualization. 
Charts to derive insights and determine which time/day-range (morning, late morning, afternoon, night) has the most accuracy. 

Also, there is a daily monitoring stack chart to visualize the breakdown of the prediction accuracy based on total number of correct, wrong and no change.  Of course, there is a slicer to adjust how we wanted to view the breakdown - daily, monthly or yearly. 

Performance metric measurement includes:
i) RMSE (not shown in powerBI)
ii) Accuracy percentage (shown in powerBI)
iii) Accuracy percentage based on time range (shown in powerBI). 

I have provided a copy of my script. For the API key, please retrieve it from Alpha vantage https://www.alphavantage.co/. It is free but you have to sign up with your email.

With reference from: 
https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html

If you have comments and feedback, pls drop me a an email at emilysmtan@hotmail.com. 

Disclaimer: The stock market is pretty volatile. This machine learning definitely has potential for being used but as far as it is concerned, it is only experimental and I wouldn't really recommend to make decisionsm merely based on a simple framework.
