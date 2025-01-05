import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to fetch cryptocurrency data
def fetch_data(crypto_symbol, start_date, end_date):
    data = yf.download(crypto_symbol, start=start_date, end=end_date)
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%d %b %Y')  # Format date
    return data

# Function to preprocess data
def preprocess_data(df):
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Close'].rolling(window=50).std()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df

# Function to prepare data for LSTM
def prepare_lstm_data(df, lookback=60):
    data = df[['SMA_50', 'SMA_200', 'Volatility']].values
    target = df['Log_Returns'].values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future values
def predict_future_lstm(model, last_sequence, steps=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(steps):
        pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        future_predictions.append(pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = pred[0, 0]
    return future_predictions

# Streamlit app interface
def main():
    st.title("Cryptocurrency Price Prediction with LSTM")
    st.markdown("""
    Predict cryptocurrency prices using LSTM models based on historical data.
    Select a cryptocurrency, specify a date range, and see predictions for future prices!
    """)

    st.sidebar.header("User Inputs")
    st.sidebar.markdown(
        '<p style="color:red; font-weight:bold;">Note: Updating the date on the side panel will re-run the algorithm.</p>',
        unsafe_allow_html=True
    )
    coins = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "SOL-USD"]
    use_custom_input = st.sidebar.checkbox("Manually Input Cryptocurrency Symbol")

    if use_custom_input:
        selected_coin = st.sidebar.text_input("Enter Cryptocurrency Symbol", value="BTC-USD")
    else:
        selected_coin = st.sidebar.selectbox("Choose Cryptocurrency", coins)

    start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input('End Date', datetime.date.today())

    if start_date > end_date:
        st.error('Start date must be before end date.')
        return

    st.markdown("### Fetching Data")
    try:
        df = fetch_data(selected_coin, start_date, end_date)
        df = preprocess_data(df)
        st.write(f"Fetched {len(df)} records for {selected_coin}.")
    except Exception as e:
        st.error(f"Failed to fetch data for {selected_coin}. Please check the symbol and try again.")
        return

    st.markdown("### Preparing Data for LSTM")
    lookback = 60
    X, y = prepare_lstm_data(df, lookback=lookback)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    st.markdown("### Training the LSTM Model")
    st.markdown("**Note:** This step takes approximately 90 seconds.")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    st.success("Model training completed!")

    st.markdown("### Model Evaluation")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.metric("Mean Absolute Error", f"{mae:.5f}")
    st.metric("Root Mean Square Error", f"{rmse:.5f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Actual', color='blue')
    ax.plot(y_pred, label='Predicted', linestyle='--', color='orange')
    ax.set_title('Actual vs Predicted Log Returns')
    ax.set_ylabel('Log Returns')
    ax.set_xlabel('Time (Days)')  # Add units for x-axis
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Future Price Prediction")
    steps = st.sidebar.slider('Days to Predict into Future', 1, 90, 30)
    last_sequence = X_test[-1]
    future_predictions = predict_future_lstm(model, last_sequence, steps=steps)

    # Correcting the mismatch by ensuring we start from the last known close price
    last_close = df['Close'].iloc[-1]
    if isinstance(last_close, (pd.Series, np.ndarray)):
        last_close = last_close.iloc[-1] if isinstance(last_close, pd.Series) else last_close[0]

    # Predict future prices and dates
    last_date = pd.to_datetime(df['Date'].iloc[-1], format='%d %b %Y')
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps).strftime('%d %b %Y')
    cumulative_returns = np.cumsum(future_predictions)

    # Calculate the future prices from the cumulative returns
    future_prices = last_close * np.exp(cumulative_returns)

    # Creating a DataFrame for the predicted prices
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price (USD)': np.round(future_prices, 2)
    })

    st.dataframe(future_df)

    # Plotting the future predicted prices
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(future_df['Date'], format='%d %b %Y'), future_df['Predicted Price (USD)'], label='Predicted Prices', color='red', linestyle='--')
    ax.set_title('Future Price Prediction')
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Date')  # Ensure label is clear
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    # Format y-axis as currency with cents
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.2f}'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Incorporating Fear and Greed Index as an additional graph
    st.markdown("### Fear and Greed Index Impact")
    # Simulated Fear and Greed index data (replace with real data as needed)
    fear_greed_index = np.linspace(0.1, 1.0, len(df))  # Simulated range
    df['Fear_Greed_Index'] = fear_greed_index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(df['Date'], format='%d %b %Y'), df['Fear_Greed_Index'], label='Fear and Greed Index', color='green')
    ax.set_title('Fear and Greed Index Over Time')
    ax.set_ylabel('Index Value')
    ax.set_xlabel('Date')  # Ensure label is clear
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Footer Section
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 14px; color: gray; margin-top: 20px;">
            Developed by <strong>Ryan Baertlein</strong><br>
            For inquiries, reach out at <a href="mailto:rdaniel7077@gmail.com">rdaniel7077@gmail.com</a><br>
            Documentation: <a href="https://github.com/rdaniel58/Documentation/blob/main/Readme%20-%20Cryptocurrency%20LSTM%20Machine%20Learning%20Website%20by%20Ryan%20Baertlein.pdf" target="_blank">How the Website Works</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
