import streamlit as st
import numpy as np
import pickle
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Boston House Price Prediction App")
st.markdown("""
Welcome to the **Boston Housing Price Predictor**!  
Enter property details below to estimate the housing price using a trained **Decision Tree Regressor** model.
---
""")

# ---------------------------------------------------------
# Load Model Safely
# ---------------------------------------------------------
model_path = "rg_pickle.pkl"

if not os.path.exists(model_path):
    st.warning("⚠️ Model file `rg_pickle.pkl` not found. Please upload it below.")
    uploaded_file = st.file_uploader("Upload your trained model (.pkl file)", type=["pkl"])
    if uploaded_file:
        model = pickle.load(uploaded_file)
        st.success("✅ Model loaded successfully from uploaded file.")
    else:
        st.stop()  # Stop the app until model is uploaded
else:
    with open(model_path, "rb") as file:
        model = pickle.load(file)

# ---------------------------------------------------------
# Input Fields
# ---------------------------------------------------------
st.header("📊 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    CRIM = st.number_input("CRIM (Per capita crime rate)", min_value=0.0, value=0.1)
    ZN = st.number_input("ZN (Residential land zoned)", min_value=0.0, value=0.0)
    INDUS = st.number_input("INDUS (Non-retail business acres)", min_value=0.0, value=5.0)
    CHAS = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
    NOX = st.number_input("NOX (Nitric oxides concentration)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

with col2:
    RM = st.number_input("RM (Avg number of rooms per dwelling)", min_value=0.0, value=6.0)
    AGE = st.number_input("AGE (% built before 1940)", min_value=0.0, value=60.0)
    DIS = st.number_input("DIS (Distance to employment centers)", min_value=0.0, value=4.0)
    RAD = st.number_input("RAD (Accessibility to highways)", min_value=1, value=1)
    TAX = st.number_input("TAX (Property tax rate)", min_value=0, value=300)

with col3:
    PTRATIO = st.number_input("PTRATIO (Pupil-teacher ratio)", min_value=0.0, value=18.0)
    B = st.number_input("B (1000(Bk - 0.63)^2)", min_value=0.0, value=350.0)
    LSTAT = st.number_input("LSTAT (% lower status of population)", min_value=0.0, value=12.0)

# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------
if st.button("🔍 Predict House Price"):
    try:
        input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD,
                                TAX, PTRATIO, B, LSTAT]])
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 **Predicted House Price:** ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
---
📦 **Model used:** Decision Tree Regressor  
📁 **Pickle file:** rg_pickle.pkl  
👨‍💻 *Developed by Pranav*
""")
