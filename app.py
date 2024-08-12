import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('label_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# Sidebar for user input
st.sidebar.header('Customer Details')
geography = st.sidebar.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92)
balance = st.sidebar.number_input('Balance')
credit_score = st.sidebar.number_input('Credit Score')
estimated_salary = st.sidebar.number_input('Estimated Salary')
tenure = st.sidebar.slider('Tenure', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Display inputs on main page
st.subheader('Customer Information')
st.write(f"**Geography:** {geography}")
st.write(f"**Gender:** {gender}")
st.write(f"**Age:** {age}")
st.write(f"**Balance:** {balance}")
st.write(f"**Credit Score:** {credit_score}")
st.write(f"**Estimated Salary:** {estimated_salary}")
st.write(f"**Tenure:** {tenure}")
st.write(f"**Number of Products:** {num_of_products}")
st.write(f"**Has Credit Card:** {has_cr_card}")
st.write(f"**Is Active Member:** {is_active_member}")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the results dynamically
st.subheader('Prediction Results')
st.write(f'**Churn Probability:** {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')

# Explain the prediction (optional)
st.markdown("""
*Factors influencing the prediction include credit score, account balance, number of products, tenure, and membership status.*
""")

# Optional: Add feedback mechanism or additional features
st.sidebar.markdown('---')
st.sidebar.header('Additional Features')
