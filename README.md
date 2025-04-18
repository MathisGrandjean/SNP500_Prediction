# 📈 S&P 500 Price Movement Prediction Using LSTM Neural Networks

This project implements a machine learning system designed to **predict whether the S&P 500 index will rise or fall on the next trading day**. It uses **Long Short-Term Memory (LSTM)** neural networks—an architecture well-suited for time series forecasting—to analyze historical price data and technical indicators for binary market direction predictions.

---

## 🧠 Project Overview

The system collects historical stock market data for the **S&P 500 (via the SPY ETF)** using the **Alpaca trading API**, processes this data into meaningful features, trains an LSTM model, and predicts whether the market will go **up or down** the next day. 

The model provides:
- A **binary prediction** (1 = up, 0 = down)
- A **probability estimate** indicating confidence

---

## ⚙️ Technical Components

### 📊 Data Collection & Processing

Handled by the `Data_import_Alpaca` class:
- Connects to the **Alpaca API** using authentication keys
- Retrieves daily OHLCV bars for **SPY**
- Formats and cleans data for analysis
- Default range: **January 1980 to present**
- Includes error handling for API requests

---

### 🧪 Feature Engineering

Managed by the `features_development` class, which transforms raw price data into technical indicators, including:

- Daily Returns
- RSI (14-day)
- MACD
- Simple Moving Averages (50, 200 days)
- ATR (14-day)
- Bollinger Bands %B
- Momentum (5, 10, 20 days)
- Volatility (5, 10, 20 days)

**Target variable:** Binary  
`1` if next day’s close > today’s close, `0` otherwise.

---

### 🧱 Model Architecture & Training

Implemented in the `LSTMPredictor` class using **TensorFlow/Keras**:

#### 🔄 Data Preprocessing:
- Feature normalization (`MinMaxScaler`)
- Time series windowing (default: 10 days)
- 80/20 train-test split

#### 🧠 Neural Network Structure:
- LSTM layer (50 units, return sequences)
- Dropout (20%)
- LSTM layer (50 units)
- Dropout (20%)
- Dense output layer with **sigmoid** activation

#### ⚙️ Training:
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Epochs**: 50
- **Batch size**: 32
- 20% validation split during training

---

### 📈 Prediction & Evaluation

Post-training, the model:
- Predicts on test set to evaluate accuracy
- Outputs metrics for both training and test data
- Uses most recent 10 days of data for **next-day forecast**
- Returns:
  - Binary prediction (up/down)
  - Probability/confidence score

This gives users **actionable insights** with confidence levels to support potential trading decisions.

---

## 🗂️ Project Structure

The codebase is organized into modular components:

main.py: Orchestrates the entire pipeline

data_alpaca.py: Handles data retrieval and cleaning

features_development.py: Generates technical indicators

model.py: Implements the LSTM neural network and prediction logic

This structure enables easy maintenance and potential expansion of the system with additional features or alternative model architectures.
