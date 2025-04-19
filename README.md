#  S&P 500 Price Movement Prediction Using LSTM Neural Networks

This projet implements a machine learning model designed to **predit whether the SNP500 will rise or fall on the next day**. To do so, it uses the **Long Short-Term Memory (LSTM)** neural network that is well-suited for times series predictions


##  Project Overview

The market data of the **SNP500** are collected using the **Alpace API** and then transformed into meaningfuL features. The LSTM model is then trained to predicts the direction fo the market. At the end, the model provides a binary predictions of the direction (1=up, 0= down) and a probibaility estimations of the the direction.



##  Technical Components

###  Data Collection & Processing

The 'Data_import_Alpaca' Class are used to import data and to clean them. First, we connect to the Alpaca API thanks to our keys. Then, it retrieves the daily bars for "SPY". The data are then stack in a dataframe and clean to be used for features engineering. The default range of date is from January 1980 until now. At the end, we get a dataframe of the data of the SNP500.


###  Feature Engineering


The features engineering is developed by the 'features_developmen' class, which tranform raw price data into technical indicator as :

- Daily Returns (percentage change between the current value and the previous value
- RSI (14-day) : 100-(100/1+RS) with RS = average gain on 14 days/ average loss on 14 days
- MACD : EMA 12(P) -EMA26 (P) with AMA n(P) the exponential moving average over n days 
- Simple Moving Averages (50, 200 days) : simple moving average on the n days
- ATR (14-day) : 14 days moving average of the TR(t)
- Bollinger Bands
- Momentum (5, 10, 20 days) : Price at t - price at t-n
- Volatility (5, 10, 20 days) 

The target variable is a binary variable equal to 1 if next day’s close > today’s close or 0 otherwise.

---

###  Model Architecture & Training

The LSTM model is contained in the 'LSTMPredictor class. The model uses TensorFlow/Keras.
The LSTM model is a kind of neural network that is really god at remembering pattern. 
First, we need to process data.

####  Data Preprocessing:

The function 'preprocess data' starts by normalizing the features. All features are scaled between 0 and 1. The data are then sequenced by group. The size is defined by time_steps. Here, the time steps is 50 so, the model used the 50 past days to predict the future value.
Finally, we split the dataset : 80% for training and 20% for testing

####  Neural Network Structure:
The neural network is then constructuted

The first layer has 50 units. Then we drop 20% of the neurons to prevent overfitting. 
The second layer contains also 50 units and we drop again 20% of the neurons. The output layer it one neuron for binary outuput (up or down) and we used the sigmoid as activation functions that output a value between 0 and 1 which gives us a probability


#### Training:

The model used the adam optimizer and the loss fonction is the binary cross entropy because the model adresses binary classification problem. The accuracy evaluate the correctness of the prediction.
The model will go trough the entire training data 50 times (epochs=50) and the model updates it weights every 32 samples. Finally, 20% of the training data is used to check for overfitting during the training. 



---

### Prediction & Evaluation

The function 'predict_next_day' is used for predicting the value of the next day. The function is the 50 last days and reshaped the data. Then, it provides the output that gives us the direcition of the next day and the probability.
finally, we calculate the accuracy of our model on the training set and the test set.

At the end, the user will have the future of the direction of the SNP500 and a probability. 


---

##  Results

On average, when the model is used, we can see that the accuracy range between 0.55 and 0.6.


