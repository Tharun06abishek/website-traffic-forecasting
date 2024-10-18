import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load the dataset
  # Update to the correct path
web = pd.read_csv("website_wata.csv")

# Encode the 'Traffic Source' column
label_encoder = LabelEncoder()
web['Traffic_source_encoded'] = label_encoder.fit_transform(web['Traffic Source'])

# Define features and target
X = web[['Page Views', 'Session Duration', 'Bounce Rate', 'Traffic_source_encoded',
         'Time on Page', 'Previous Visits']]
y = web['Conversion Rate']

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit User Interface
st.title("Website Conversion Rate Prediction")

st.write("This app predicts the conversion rate based on user inputs for various metrics related to web traffic.")

# Collecting user input
page_views = st.number_input("Page Views", min_value=1, max_value=100, value=5)
session_duration = st.number_input("Session Duration (minutes)", min_value=0.1, max_value=100.0, value=5.0)
bounce_rate = st.number_input("Bounce Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
traffic_source = st.selectbox("Traffic Source", options=['Organic', 'Paid', 'Social', 'Referral', 'Direct'])
time_on_page = st.number_input("Time on Page (minutes)", min_value=0.1, max_value=100.0, value=5.0)
previous_visits = st.number_input("Previous Visits", min_value=0, max_value=100, value=3)

# Encode the 'Traffic Source' input
traffic_source_encoded = label_encoder.transform([traffic_source])[0]

# Create DataFrame for the new input
new_data = pd.DataFrame([[page_views, session_duration, bounce_rate,
                          traffic_source_encoded, time_on_page, previous_visits]],
                        columns=X.columns)

# Predict the conversion rate
predicted_conversion_rate = model.predict(new_data)[0]

# Display the predicted conversion rate
st.write(f"Predicted Conversion Rate: {predicted_conversion_rate:.2f}")

# Plotting predicted vs. actual conversion rates for the last 5 samples in the dataset
import matplotlib.pyplot as plt
import numpy as np

# Prepare for plotting:
last_five_actual = y.tail(5).values  # Last 5 actual conversion rates
predicted_values = np.append(last_five_actual, predicted_conversion_rate)  # Add predicted value

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), last_five_actual, marker='o', label='Actual Conversion Rate')
plt.plot(6, predicted_conversion_rate, marker='o', color='red', label='Predicted Value')
# Labels and Legends
plt.xlabel('Sample')
plt.ylabel('Conversion Rate')
plt.title('Actual vs Predicted Conversion Rate')
plt.legend()

# Show plot in Streamlit
st.pyplot(plt)
