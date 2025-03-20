import streamlit as st
import numpy as np
import joblib

# Load the trained GradientBoostingRegressor model
@st.cache_data()
def load_model():
    return joblib.load('linear_regression_model.pkl')  # Load your trained model here

gb_regressor = load_model()

# Function to predict insurance charges
def predict_insurance_charges(input_data):
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = gb_regressor.predict(input_data_reshaped)
    return prediction[0]

# Streamlit App
def main():
    st.title('Insurance Charge Prediction')

    # Input form
    st.sidebar.header('Input Parameters')
    age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=30)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    smoker = st.sidebar.selectbox('Smoker', ['Yes', 'No'])
    children = st.sidebar.number_input('Number of Children', min_value=0, max_value=10, value=0)
    region = st.sidebar.selectbox('Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])

    # Map categorical inputs to numerical values
    sex_map = {'Male': 0, 'Female': 1}
    smoker_map = {'Yes': 0, 'No': 1}
    region_map = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}

    # Predict insurance charges when the button is clicked
    if st.sidebar.button('Predict'):
        # Predict insurance charges
        input_data = (age, sex_map[sex], bmi, smoker_map[smoker], children, region_map[region])
        predicted_charges = predict_insurance_charges(input_data)

        # Display result
        st.write('### Predicted Insurance Charges:')
        st.write(f'INR {predicted_charges:.2f}')

if __name__ == '__main__':
    main()
