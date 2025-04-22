import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMPredictor:
    def __init__(self, X, y, time_steps=10):
        self.X = X
        self.y = y
        self.time_steps = time_steps
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self):
        X_scaled = self.scaler.fit_transform(self.X)

        self.X_processed, self.y_processed = [], []
        for i in range(self.time_steps, len(X_scaled)):
            self.X_processed.append(X_scaled[i-self.time_steps:i, :])
            self.y_processed.append(self.y[i])
        self.X_processed, self.y_processed = np.array(self.X_processed), np.array(self.y_processed)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_processed, self.y_processed, test_size=0.2, random_state=42)

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=50, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)


    def predict_next_day(self):
        last_data = self.X.tail(self.time_steps).values
        last_data_scaled = self.scaler.transform(last_data)
        last_data_scaled = np.reshape(last_data_scaled, (1, last_data_scaled.shape[0], last_data_scaled.shape[1]))
        prediction = self.model.predict(last_data_scaled)[0][0]
        print(prediction)
        if prediction > 0.5:
            print(f"Le S&P 500 va monter demain avec une probabilité de {prediction:.2%}")

        else:
            print(f"Le S&P 500 va descendre demain avec une probabilité de {1-prediction:.2%}")

    def calculate_accuracy(self):
        
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        

        train_predictions = (train_predictions > 0.5).astype(int)
        test_predictions = (test_predictions > 0.5).astype(int)

       
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        print(f'Train Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}')
        return train_accuracy, test_accuracy

