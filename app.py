import streamlit as st
import pickle
import numpy as np

# ---------------------------------------------------------
# Load the trained Decision Tree model
# ---------------------------------------------------------
with open("best_pickle.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------------------------------------------------
# App Title and Description
# ---------------------------------------------------------
st.title("🏠 Boston House Price Prediction App")
st.markdown("""
### Predict House Price using Decision Tree Regression  
Enter the property details below to estimate its price.
""")

# ---------------------------------------------------------
# Input Features
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    CRIM = st.number_input("CRIM (Per capita crime rate)", min_value=0.0)
    ZN = st.number_input("ZN (Residential land zoned)", min_value=0.0)
    INDUS = st.number_input("INDUS (Non-retail business acres)", min_value=0.0)
    CHAS = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
    NOX = st.number_input("NOX (Nitric oxides concentration)", min_value=0.0, max_value=1.0, step=0.01)

with col2:
    RM = st.number_input("RM (Average number of rooms per dwelling)", min_value=0.0)
    AGE = st.number_input("AGE (Proportion of owner-occupied units built before 1940)", min_value=0.0)
    DIS = st.number_input("DIS (Weighted distances to employment centers)", min_value=0.0)
    RAD = st.number_input("RAD (Accessibility to radial highways)", min_value=0)
    TAX = st.number_input("TAX (Full-value property-tax rate per $10,000)", min_value=0)

with col3:
    PTRATIO = st.number_input("PTRATIO (Pupil-teacher ratio by town)", min_value=0.0)
    B = st.number_input("B (1000(Bk - 0.63)^2)", min_value=0.0)
    LSTAT = st.number_input("LSTAT (% lower status of the population)", min_value=0.0)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔍 Predict Price"):
    try:
        # Prepare input for prediction
        features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        st.success(f"🏡 **Estimated House Price: ${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
---
📊 *Model: Decision Tree Regressor (best_pickle.pkl)*  
💡 *Created by: Pranav*
""")
