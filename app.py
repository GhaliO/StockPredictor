import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    file_path = "MacroTrends_Data_Download_AAPL.xlsx"
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip().str.lower()  # Standardize column names to lowercase
    data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' is in datetime format
    data = data.sort_values(by='date')  # Sort by date
    return data

# Load data
data = load_data()

# Prepare the data
data['previous_close'] = data['close'].shift(1)  # Add 'Previous Close' column
data = data.dropna()  # Remove rows with missing values

# Define features and target variables
X = data[['previous_close', 'open']].reset_index(drop=True)  # Simplified feature set
y_high = data['high'].reset_index(drop=True)
y_low = data['low'].reset_index(drop=True)
y_close = data['close'].reset_index(drop=True)
dates = data['date'].reset_index(drop=True)  # Reset index for dates

# Split the data into training (70%) and testing (30%)
X_train, X_test, y_high_train, y_high_test, dates_train, dates_test = train_test_split(
    X, y_high, dates, test_size=0.3, random_state=42)

_, _, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.3, random_state=42)
_, _, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.3, random_state=42)

# Filter test dates between 2014 and 2024
test_data_mask = (dates_test >= pd.to_datetime("2014-01-01")) & (dates_test <= pd.to_datetime("2024-12-31"))
X_test = X_test[test_data_mask]
dates_test = dates_test[test_data_mask]
y_high_test = y_high_test[test_data_mask]
y_low_test = y_low_test[test_data_mask]
y_close_test = y_close_test[test_data_mask]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Ridge Regression models
ridge_high = Ridge(alpha=1.0)
ridge_low = Ridge(alpha=1.0)
ridge_close = Ridge(alpha=1.0)

# Train models
ridge_high.fit(X_train_scaled, y_high_train)
ridge_low.fit(X_train_scaled, y_low_train)
ridge_close.fit(X_train_scaled, y_close_train)

# Predict on test data
predicted_high = ridge_high.predict(X_test_scaled)
predicted_low = ridge_low.predict(X_test_scaled)
predicted_close = ridge_close.predict(X_test_scaled)

# Create a DataFrame for results
results = pd.DataFrame({
    'date': dates_test,
    'predicted_high': predicted_high,
    'actual_high': y_high_test.values,
    'predicted_low': predicted_low,
    'actual_low': y_low_test.values,
    'predicted_close': predicted_close,
    'actual_close': y_close_test.values
}).sort_values(by="date")

# Today's prediction
latest_data = data.iloc[-1][['previous_close', 'open']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
today_high_pred = ridge_high.predict(latest_data_scaled)[0]
today_low_pred = ridge_low.predict(latest_data_scaled)[0]
today_close_pred = ridge_close.predict(latest_data_scaled)[0]

# Streamlit Interface
st.title("Stock Price Prediction for AAPL (2014-2024 Testing Set)")

# Display today's predictions
st.write("### Today's Predicted Prices")
st.metric(label="Predicted High", value=f"${today_high_pred:.2f}")
st.metric(label="Predicted Low", value=f"${today_low_pred:.2f}")
st.metric(label="Predicted Close", value=f"${today_close_pred:.2f}")

# Dropdown and input for date selection
st.write("### Search Stock Data by Date")
search_date = st.text_input("Enter a date (YYYY-MM-DD):", "")
selected_date = st.selectbox("Or select a date:", options=results['date'].dt.strftime('%Y-%m-%d'))

# Display results for the selected date
if search_date:
    try:
        search_date = pd.to_datetime(search_date)
        selected_row = results[results['date'] == search_date]
        if not selected_row.empty:
            st.write(f"### Results for {search_date.strftime('%Y-%m-%d')}")
            for _, row in selected_row.iterrows():
                st.write(f"**Predicted High:** ${row['predicted_high']:.2f}")
                st.write(f"**Actual High:** ${row['actual_high']:.2f}")
                st.write(f"**Predicted Low:** ${row['predicted_low']:.2f}")
                st.write(f"**Actual Low:** ${row['actual_low']:.2f}")
                st.write(f"**Predicted Close:** ${row['predicted_close']:.2f}")
                st.write(f"**Actual Close:** ${row['actual_close']:.2f}")
        else:
            st.error("No data found for the entered date.")
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
elif selected_date:
    selected_row = results[results['date'].dt.strftime('%Y-%m-%d') == selected_date]
    st.write(f"### Results for {selected_date}")
    for _, row in selected_row.iterrows():
        st.write(f"**Predicted High:** ${row['predicted_high']:.2f}")
        st.write(f"**Actual High:** ${row['actual_high']:.2f}")
        st.write(f"**Predicted Low:** ${row['predicted_low']:.2f}")
        st.write(f"**Actual Low:** ${row['actual_low']:.2f}")
        st.write(f"**Predicted Close:** ${row['predicted_close']:.2f}")
        st.write(f"**Actual Close:** ${row['actual_close']:.2f}")

# Line chart comparing predicted and actual close prices
st.write("### Comparison of Predicted vs. Actual Close Prices")
filtered_results = results[(results['date'] >= "2014-01-01") & (results['date'] <= "2024-12-31")]
st.line_chart(filtered_results.set_index('date')[['predicted_close', 'actual_close']])
