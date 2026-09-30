# 📈 Stock Price Prediction using ML & Neural Networks (LSTM)

An interactive, production-ready Streamlit web application powered by Deep Learning (LSTM Recurrent Neural Networks) to analyze and predict stock price trends using real-time market data from Yahoo Finance.

---

## 🌟 Key Features

- **Real-Time Market Data**: Pulls live historical price and volume data using anti-throttled Yahoo Finance streams.
- **Moving Average Technical Analysis**: Rolling trend analysis for 100-day, 200-day, and 250-day moving averages.
- **Deep Learning Forecasting**: Multi-layer LSTM neural network trained to forecast test stock prices based on a 100-day rolling lookback window.
- **Performance Evaluation**: Real-time calculation of prediction metrics including **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and **MAPE** (Mean Absolute Percentage Error).
- **Interactive UI & Dark Mode**: Sleek financial dashboard with KPI metric cards, tabs, and exportable CSV data.

---

## 🏗️ Neural Network Architecture

- **Model Type**: Sequential LSTM Deep Neural Network
- **Layer 1**: LSTM (128 units, `return_sequences=True`)
- **Layer 2**: LSTM (64 units, `return_sequences=False`)
- **Layer 3**: Dense (25 units, ReLU)
- **Output Layer**: Dense (1 unit, Linear)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
