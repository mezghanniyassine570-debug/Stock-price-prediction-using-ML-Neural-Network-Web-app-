import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from curl_cffi import requests
try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0b0f19;
    }
    
    /* Header hero styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }
    
    .metric-delta-pos {
        color: #10B981;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .metric-delta-neg {
        color: #EF4444;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Section card headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        color: #E2E8F0;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.6);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions with caching for performance and reliability
@st.cache_resource
def get_lstm_model():
    model_path = "Latest_stock_price_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found.")
        return None
    return load_model(model_path)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data(ticker_symbol: str, years: int = 20):
    end_date = datetime.now()
    start_date = datetime(end_date.year - years, end_date.month, end_date.day)
    
    try:
        # Use curl_cffi session to bypass Yahoo Finance rate limits
        session = requests.Session(impersonate="chrome")
        ticker = yf.Ticker(ticker_symbol, session=session)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            # Fallback to standard yfinance
            df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
        # Clean index timezone if present for clean plotting
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker_symbol}: {e}")
        return pd.DataFrame()

# Matplotlib dark theme configuration
def setup_plt_style():
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']
    plt.rcParams['figure.facecolor'] = '#0B0F19'
    plt.rcParams['axes.facecolor'] = '#0E1424'
    plt.rcParams['axes.edgecolor'] = '#1E293B'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#1E293B'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'

setup_plt_style()

# Sidebar controls
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.header("⚙️ Configuration")
    
    # Pre-set quick picks
    popular_stocks = ["GOOG", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
    selected_quick = st.selectbox("Quick Select Stock", ["Custom"] + popular_stocks, index=1)
    
    if selected_quick != "Custom":
        default_stock = selected_quick
    else:
        default_stock = "GOOG"
        
    stock = st.text_input("Enter the Stock ID", value=default_stock).upper().strip()
    years_lookback = st.slider("Historical Data Range (Years)", min_value=1, max_value=25, value=20)
    
    st.divider()
    st.markdown("### 🧠 Model Info")
    st.info("**Architecture**: 2x LSTM layers (128 & 64 units) + Dense(25) + Dense(1)\n\n**Sequence Length**: 100 days\n**Target**: Close Price")

# Hero Banner
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">📈 Stock Price Predictor AI</div>
    <div class="hero-subtitle">Real-time market analytics, rolling trend analysis, and LSTM Deep Learning predictions for <b>{stock}</b></div>
</div>
""", unsafe_allow_html=True)

# Main Application Flow
with st.spinner(f"Fetching real-time historical data for {stock}..."):
    google_data = fetch_stock_data(stock, years=years_lookback)

if google_data.empty or len(google_data) < 150:
    st.warning(f"Could not retrieve sufficient trading data for ticker `{stock}`. Please enter a valid ticker symbol (e.g. GOOG, AAPL, MSFT).")
    st.stop()

# Ensure Close column exists
if 'Close' not in google_data.columns:
    if 'Adj Close' in google_data.columns:
        google_data['Close'] = google_data['Adj Close']
    else:
        st.error("Close price column missing from dataset.")
        st.stop()

# Top KPI Summary Row
latest_close = float(google_data['Close'].iloc[-1])
prev_close = float(google_data['Close'].iloc[-2])
change_abs = latest_close - prev_close
change_pct = (change_abs / prev_close) * 100
high_52w = float(google_data['Close'].tail(252).max())
low_52w = float(google_data['Close'].tail(252).min())
vol_latest = int(google_data['Volume'].iloc[-1]) if 'Volume' in google_data.columns else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    delta_class = "metric-delta-pos" if change_abs >= 0 else "metric-delta-neg"
    sign = "+" if change_abs >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest Close Price</div>
        <div class="metric-val">${latest_close:,.2f}</div>
        <div class="{delta_class}">{sign}${change_abs:,.2f} ({sign}{change_pct:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">52-Week High</div>
        <div class="metric-val">${high_52w:,.2f}</div>
        <div style="color: #94A3B8; font-size: 0.85rem;">Peak in 252 trading days</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">52-Week Low</div>
        <div class="metric-val">${low_52w:,.2f}</div>
        <div style="color: #94A3B8; font-size: 0.85rem;">Trough in 252 trading days</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Trading Volume</div>
        <div class="metric-val">{vol_latest:,.0f}</div>
        <div style="color: #94A3B8; font-size: 0.85rem;">Latest Daily Volume</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Helper function for plotting graphs with rich aesthetic
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, val_label="Moving Average", extra_label=""):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    # Plot Original Close Price
    ax.plot(full_data.index, full_data['Close'], label=f'{stock} Close Price', color='#38BDF8', linewidth=1.8, alpha=0.9)
    
    # Plot MA
    if values is not None:
        ax.plot(full_data.index, values, label=val_label, color='#F59E0B', linewidth=2.0)
        
    # Plot Extra MA if provided
    if extra_data and extra_dataset is not None:
        ax.plot(full_data.index, extra_dataset, label=extra_label if extra_label else "Extra MA", color='#EC4899', linewidth=2.0)
        
    ax.set_title(f"{stock} Historical Price & Trend Indicators", fontsize=14, fontweight='bold', pad=12, color='#F8FAFC')
    ax.set_xlabel("Date", fontsize=11, labelpad=8, color='#94A3B8')
    ax.set_ylabel("Price (USD)", fontsize=11, labelpad=8, color='#94A3B8')
    ax.legend(loc="upper left", frameon=True, facecolor="#111827", edgecolor="#374151", fontsize=10)
    ax.tick_params(colors='#94A3B8', which='both')
    plt.tight_layout()
    return fig

# Tabbed Interface for structured exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Moving Average Analysis",
    "🤖 LSTM AI Prediction",
    "📋 Raw Market Data",
    "📈 Interactive Chart"
])

# Compute Moving Averages
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()

with tab1:
    st.markdown('<div class="section-header">📉 Moving Average Trend Breakdown</div>', unsafe_allow_html=True)
    
    ma_choice = st.radio(
        "Select Moving Average Window to Inspect:",
        ["All Overlaid (100 vs 250 Days)", "100-Day Moving Average", "200-Day Moving Average", "250-Day Moving Average"],
        horizontal=True
    )
    
    if ma_choice == "All Overlaid (100 vs 250 Days)":
        st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
        fig = plot_graph(
            (14, 5.5),
            google_data['MA_for_100_days'],
            google_data,
            extra_data=1,
            extra_dataset=google_data['MA_for_250_days'],
            val_label="100-Day MA (Short/Mid Trend)",
            extra_label="250-Day MA (Long Term Trend)"
        )
        st.pyplot(fig)
    elif ma_choice == "100-Day Moving Average":
        st.subheader('Original Close Price and MA for 100 days')
        fig = plot_graph(
            (14, 5.5),
            google_data['MA_for_100_days'],
            google_data,
            val_label="100-Day MA"
        )
        st.pyplot(fig)
    elif ma_choice == "200-Day Moving Average":
        st.subheader('Original Close Price and MA for 200 days')
        fig = plot_graph(
            (14, 5.5),
            google_data['MA_for_200_days'],
            google_data,
            val_label="200-Day MA"
        )
        st.pyplot(fig)
    else:
        st.subheader('Original Close Price and MA for 250 days')
        fig = plot_graph(
            (14, 5.5),
            google_data['MA_for_250_days'],
            google_data,
            val_label="250-Day MA"
        )
        st.pyplot(fig)

with tab2:
    st.markdown('<div class="section-header">🤖 Deep Learning (LSTM) Model Inference</div>', unsafe_allow_html=True)
    
    model = get_lstm_model()
    
    if model is not None:
        splitting_len = int(len(google_data) * 0.7)
        x_test_df = pd.DataFrame(google_data['Close'].iloc[splitting_len:])
        
        if len(x_test_df) <= 100:
            st.error("Insufficient test data points. Increase the lookback years in the sidebar.")
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(x_test_df[['Close']])
            
            x_data = []
            y_data = []
            
            for i in range(100, len(scaled_data)):
                x_data.append(scaled_data[i-100:i])
                y_data.append(scaled_data[i])
                
            x_data, y_data = np.array(x_data), np.array(y_data)
            
            with st.spinner("Generating LSTM predictions..."):
                predictions = model.predict(x_data, verbose=0)
                
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)
            
            ploting_data = pd.DataFrame(
                {
                    'original_test_data': inv_y_test.reshape(-1),
                    'predictions': inv_pre.reshape(-1)
                },
                index=google_data.index[splitting_len + 100:]
            )
            
            # Prediction evaluation metrics
            rmse = np.sqrt(np.mean((inv_pre - inv_y_test) ** 2))
            mae = np.mean(np.abs(inv_pre - inv_y_test))
            mape = np.mean(np.abs((inv_y_test - inv_pre) / inv_y_test)) * 100
            
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.metric("Test Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
            with pcol2:
                st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
            with pcol3:
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
                
            st.subheader('Original Close Price vs Predicted Close Price')
            
            # High-end comparison chart
            fig_pred, ax_pred = plt.subplots(figsize=(14, 6), dpi=120)
            
            # Training / Unused historical segment
            train_idx = google_data.index[:splitting_len + 100]
            train_vals = google_data['Close'].iloc[:splitting_len + 100]
            
            ax_pred.plot(train_idx, train_vals, label="Historical Data (Train/Unused)", color='#64748B', linewidth=1.5, alpha=0.7)
            ax_pred.plot(ploting_data.index, ploting_data['original_test_data'], label="Actual Test Ground Truth", color='#10B981', linewidth=2.0)
            ax_pred.plot(ploting_data.index, ploting_data['predictions'], label="LSTM AI Predictions", color='#F59E0B', linewidth=2.0, linestyle="--")
            
            ax_pred.set_title(f"{stock} Full Sequence: Historical Baseline vs Actual vs Predicted", fontsize=14, fontweight='bold', pad=12, color='#F8FAFC')
            ax_pred.set_xlabel("Date", fontsize=11, labelpad=8, color='#94A3B8')
            ax_pred.set_ylabel("Price (USD)", fontsize=11, labelpad=8, color='#94A3B8')
            ax_pred.legend(loc="upper left", frameon=True, facecolor="#111827", edgecolor="#374151", fontsize=10)
            ax_pred.tick_params(colors='#94A3B8', which='both')
            plt.tight_layout()
            st.pyplot(fig_pred)
            
            st.subheader("Original values vs Predicted values (Recent Sample)")
            st.dataframe(ploting_data.tail(20).style.format("${:,.2f}"), use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">📋 Historical Market Data Table</div>', unsafe_allow_html=True)
    st.subheader("Stock Data")
    st.dataframe(google_data.tail(100), use_container_width=True)
    
    csv = google_data.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Historical CSV",
        data=csv,
        file_name=f"{stock}_stock_data.csv",
        mime="text/csv"
    )

with tab4:
    st.markdown('<div class="section-header">📈 Interactive Price & Volume Chart</div>', unsafe_allow_html=True)
    st.line_chart(google_data[['Close', 'MA_for_100_days', 'MA_for_250_days']])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px 0;">
        Stock Price Predictor AI App • Powered by Streamlit, TensorFlow/Keras & Yahoo Finance
    </div>
    """,
    unsafe_allow_html=True
)