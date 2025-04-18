from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
import pandas as pd
import datetime

class Data_import_Alpaca:

    def __init__(self, symbol, api_key, api_secret, start_date, end_date):
        # Vérifier si symbol est une liste et prendre le premier élément si c'est le cas
        self.symbol = symbol[0] if isinstance(symbol, list) else symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.start_date = start_date
        self.end_date = end_date

    def import_data(self):
        client = StockHistoricalDataClient(self.api_key, self.api_secret)
        request = StockBarsRequest(
            # Si vous voulez passer une liste à l'API, utilisez la liste originale
            symbol_or_symbols=[self.symbol] if not isinstance(self.symbol, list) else self.symbol,
            timeframe=TimeFrame.Day,
            start=self.start_date,
            end=self.end_date
        )
        bars = client.get_stock_bars(request).df
        return bars

    def data_cleaning(self, bars):
        # Maintenant self.symbol est une chaîne, pas une liste
        data = bars[bars.index.get_level_values(0) == self.symbol].copy()
        
        # Reset index correctement
        data.reset_index(inplace=True)
        
        # Set index
        data.set_index('timestamp', inplace=True)
        
        # Format index
        data.index = data.index.strftime('%Y-%m-%d')
        
        return data

