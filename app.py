import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    file_path = "MacroTrends_Data_Download_AAPL.xlsx"
    data = pd.read_excel(file_path)
    data.columns = ["date", "open", "high", "low", "close", "volume"]  # Rename columns
    data["date"] = pd.to_datetime(data["date"])  # Ensure 'date' is datetime
    data = data.sort_values(by="date")  # Sort by date
    return data

data = load_data()

# Add derived features
data["previous_close"] = data["close"].shift(1)  # Add previous close price
data = data.dropna()  # Drop rows with missing previous close prices

# Define features (X) and target variables (y)
X = data[["previous_close", "open"]]
y_high = data["high"]
y_low = data["low"]
y_close = data["close"]

# Split the data into training (70%) and testing (30%)
X_train, X_test, y_high_train, y_high_test = train_test_split(X, y_high, test_size=0.3, random_state=42)
_, _, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.3, random_state=42)
_, _, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Ridge Regression models for High, Low, and Close prices
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

# Filter testing data to 2014-2024 range
test_data = data.iloc[X_test.index]
test_data = test_data[(test_data["date"].dt.year >= 2014) & (test_data["date"].dt.year <= 2024)]
results = pd.DataFrame({
    "date": test_data["date"].values,
    "predicted_high": predicted_high[:len(test_data)],
    "actual_high": y_high_test.values[:len(test_data)],
    "predicted_low": predicted_low[:len(test_data)],
    "actual_low": y_low_test.values[:len(test_data)],
    "predicted_close": predicted_close[:len(test_data)],
    "actual_close": y_close_test.values[:len(test_data)]
})

# Streamlit Interface
st.title("Stock Price Prediction - Testing Set (2014-2024)")
st.write("Select a date to view predicted and actual stock prices.")

# Dropdown for selecting a date
selected_date = st.selectbox("Select a date:", options=results["date"].dt.strftime("%Y-%m-%d"))

if selected_date:
    selected_row = results[results["date"].dt.strftime("%Y-%m-%d") == selected_date]
    st.write(f"### Results for {selected_date}")
    st.dataframe(selected_row)

# Add a line chart to compare predicted vs actual close prices
st.write("### Comparison of Predicted vs Actual Close Prices")
st.line_chart(results.set_index("date")[["predicted_close", "actual_close"]])
