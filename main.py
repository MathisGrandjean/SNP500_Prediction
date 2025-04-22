from data.data_alpaca import Data_import_Alpaca
from features.features_development import features_development
from model.model import LSTMPredictor
import datetime
import matplotlib.pyplot as plt 

start_date=datetime.datetime(1980, 1, 1)
end_date=datetime.date.today()
symbols=['SPY']
api_key=''
secret_key=''

#import of data

data_downloader=Data_import_Alpaca(symbol=symbols,api_key=api_key,api_secret=secret_key,start_date=start_date,end_date=end_date)
bars=data_downloader.import_data()
data=data_downloader.data_cleaning(bars)


#creation of features based on prices, volume...

features_developer=features_development(data)
feature=features_developer.features_creation()

print(feature)

predictor=LSTMPredictor(feature.drop(columns='Target'),feature['Target'])
predictor.preprocess_data()
predictor.build_model()
predictor.train_model()
predictor.predict_next_day()
predictor.calculate_accuracy()
