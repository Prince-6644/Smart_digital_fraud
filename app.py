import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load Model and Scaler ---
try:
    with open('random_forest_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    st.sidebar.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model or scaler: {e}")
    st.stop() # Stop the app if essential files can't be loaded

# --- Feature Names (Order matters for prediction) ---
# The order of features must match the training data (X)
# From the X variable in kernel state:
# ['Transaction_Amount', 'Transaction_Type', 'Account_Balance',
#  'Merchant_Category', 'Transaction_Frequency', 'Location', 'Transaction_Hour']
feature_names = [
    "Transaction_Amount",
    "Transaction_Type",
    "Account_Balance",
    "Merchant_Category",
    "Transaction_Frequency",
    "Location",
    "Transaction_Hour"
]

# --- Mappings for Categorical Features ---
# These mappings are reconstructed based on the output of data.head() after encoding
# and by filling in the remaining unique categories from the raw dataset logically.

transaction_type_map = {
    "Debit Card": 0,
    "NetBanking": 1,
    "UPI": 2,
    "Credit Card": 3
}

merchant_category_map = {
    "Apparel": 2,
    "Electronics": 0,
    "Food": 1,
    "Healthcare": 4,
    "Jewelry": 3,
    "Transport": 5,
    "Travel": 6
}

location_map = {
    "Bangalore": 0,
    "Chennai": 1,
    "Delhi": 2,
    "Mumbai": 3,
    "Indore": 4
}


# --- Streamlit UI ---
st.set_page_config(page_title="Digital Fraud Detection", layout="centered")
st.title("Digital Fraud Detection System")

st.markdown("""
    Enter the transaction details below to predict if it's a fraudulent transaction.
""")

with st.sidebar:
    st.header("Transaction Details Input")
    transaction_amount = st.number_input("Transaction Amount", min_value=0, max_value=100000, value=25000)
    account_balance = st.number_input("Account Balance", min_value=0, max_value=100000, value=50000)
    transaction_frequency = st.number_input("Transaction Frequency", min_value=0, max_value=50, value=12)
    transaction_time_hour = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=14)

    # Categorical Inputs
    transaction_type_str = st.selectbox("Transaction Type", list(transaction_type_map.keys()), index=list(transaction_type_map.keys()).index("UPI"))
    merchant_category_str = st.selectbox("Merchant Category", list(merchant_category_map.keys()), index=list(merchant_category_map.keys()).index("Electronics"))
    location_str = st.selectbox("Location", list(location_map.keys()), index=list(location_map.keys()).index("Indore"))

    st.markdown("---")
    submitted = st.button("Predict Fraud Status")

if submitted:
    # --- Preprocessing User Input ---
    transaction_type_encoded = transaction_type_map.get(transaction_type_str)
    merchant_category_encoded = merchant_category_map.get(merchant_category_str)
    location_encoded = location_map.get(location_str)

    if any(val is None for val in [transaction_type_encoded, merchant_category_encoded, location_encoded]):
        st.error("Error: Could not encode categorical features. Please check mappings.")
    else:
        # Create input array in the correct feature order
        input_data = np.array([[
            transaction_amount,
            transaction_type_encoded,
            account_balance,
            merchant_category_encoded,
            transaction_frequency,
            location_encoded,
            transaction_time_hour
        ]])

        # Scale the input data
        try:
            scaled_input_data = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Error scaling input data: {e}. Make sure input dimensions match training data.")
            st.stop()

        # --- Make Prediction ---
        prediction = rf_model.predict(scaled_input_data)
        prediction_proba = rf_model.predict_proba(scaled_input_data)

        # --- Display Result ---
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
            st.write(f"Confidence (Fraud): {prediction_proba[0][1]*100:.2f}%")
            st.write(f"Confidence (Legitimate): {prediction_proba[0][0]*100:.2f}%")
        else:
            st.success("✅ Legitimate Transaction")
            st.write(f"Confidence (Legitimate): {prediction_proba[0][0]*100:.2f}%")
            st.write(f"Confidence (Fraud): {prediction_proba[0][1]*100:.2f}%")

        st.markdown("---")
        st.subheader("Input Details:")
        input_df = pd.DataFrame([input_data[0]], columns=feature_names)
        st.write(input_df)
