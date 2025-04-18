import pandas as pd 
import numpy as np 
import pandas_ta as ta

class features_development:
    
    def __init__(self, data_price):
         self.data_price=data_price
         
    def features_creation(self):
        df=pd.DataFrame()
        df['Return'] = self.data_price['close'].pct_change()
        df['RSI_14'] = ta.rsi(self.data_price['close'], length=14)
        df['MACD'] = ta.macd(self.data_price['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
        df['SMA_50'] = ta.sma(self.data_price['close'], length=50)
        df['SMA_200'] = ta.sma(self.data_price['close'], length=200)
        df['ATR_14'] = ta.atr(self.data_price['high'], self.data_price['low'], self.data_price['close'], length=14)
        df['Bollinger_%b'] = ta.bbands(self.data_price['close'], length=20)['BBP_20_2.0']
        
        for window in [5, 10, 20]:
            df[f'Momentum_{window}'] = self.data_price['close'].pct_change(window)
            df[f'Volatility_{window}'] = df['Return'].rolling(window).std()
        df['Target'] = (self.data_price['close'].shift(-1) > self.data_price['close']).astype(int)
        return df.dropna()
       
 