import base64
import csv
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
from datetime import datetime

# Load the dataset
data = pd.read_csv('fraud_data.csv')

# Encode categorical variables
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Encode 'Marital Status' using OneHotEncoder
encoder_marital_status = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
marital_status_encoded = encoder_marital_status.fit_transform(data[['Marital Status']])
marital_status_encoded_df = pd.DataFrame(marital_status_encoded, columns=encoder_marital_status.get_feature_names_out())

# Encode 'Claim Type' using OneHotEncoder
encoder_claim_type = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
claim_type_encoded = encoder_claim_type.fit_transform(data[['Claim Type']])
claim_type_encoded_df = pd.DataFrame(claim_type_encoded, columns=encoder_claim_type.get_feature_names_out())

# Encode 'Coverage' using OneHotEncoder
encoder_coverage = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
coverage_encoded = encoder_coverage.fit_transform(data[['Coverage']])
coverage_encoded_df = pd.DataFrame(coverage_encoded, columns=encoder_coverage.get_feature_names_out())

# Combine the original data with the encoded columns and drop unnecessary columns
data = pd.concat([data, marital_status_encoded_df, claim_type_encoded_df, coverage_encoded_df], axis=1).drop(columns=[
    'Policy Number', 'Customer ID', 'Customer Name', 'Mobile Number', 'Email Id', 'Location', 'Occupation',
    'Current Insurance', 'Claim Type', 'Marital Status', 'Coverage'
])

# Check for any non-numeric columns and convert if necessary
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

# Define features and target
X = data.drop(columns=['Fraud'])
y = data['Fraud']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=200, random_state=21)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy}')

# Save the model and encoders to files

joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(encoder_marital_status, 'encoder_marital_status.pkl')
joblib.dump(encoder_claim_type, 'encoder_claim_type.pkl')
joblib.dump(encoder_coverage, 'encoder_coverage.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')  # Save the feature names
print('Model, encoders, and feature names saved successfully')


def predict_fraud(new_data_dict):
    model = joblib.load('random_forest_model.pkl')
    encoder_marital_status = joblib.load('encoder_marital_status.pkl')
    encoder_claim_type = joblib.load('encoder_claim_type.pkl')
    encoder_coverage = joblib.load('encoder_coverage.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print('Model, encoders, and feature names loaded successfully')

    # Sample new data for prediction
    new_data_dict = {
      'Age': [new_data_dict['Age']],
      'Gender': [new_data_dict['Gender']],
      'Marital Status': [new_data_dict['Marital Status']],
      'Dependents': [new_data_dict['Dependents']],
      'Claims (3yrs)': [new_data_dict['Claims (3yrs)']],
      'Claim Type': [new_data_dict['Claim Type']],
      'Online Activity': [new_data_dict['Online Activity']],
      'Service Calls': [new_data_dict['Service Calls']],
      'Coverage': [new_data_dict['Coverage']],
      'Premium (3yrs)': [new_data_dict['Premium (3yrs)']]
    }

    new_data_df = pd.DataFrame(new_data_dict)
    new_data_df['Gender'] = new_data_df['Gender'].map({'Male': 0, 'Female': 1})
    new_marital_status_encoded = encoder_marital_status.transform(new_data_df[['Marital Status']])
    new_marital_status_encoded_df = pd.DataFrame(new_marital_status_encoded, columns=encoder_marital_status.get_feature_names_out())
    new_claim_type_encoded = encoder_claim_type.transform(new_data_df[['Claim Type']])
    new_claim_type_encoded_df = pd.DataFrame(new_claim_type_encoded, columns=encoder_claim_type.get_feature_names_out())
    new_coverage_encoded = encoder_coverage.transform(new_data_df[['Coverage']])
    new_coverage_encoded_df = pd.DataFrame(new_coverage_encoded, columns=encoder_coverage.get_feature_names_out())

    # Combine the new data with the encoded columns and drop unnecessary columns
    new_data_df = pd.concat([new_data_df.reset_index(drop=True), new_marital_status_encoded_df, new_claim_type_encoded_df, new_coverage_encoded_df], axis=1).drop(columns=['Marital Status', 'Claim Type', 'Coverage'])

    # Ensure the columns in the new data match the training data
    new_data_df = new_data_df.reindex(columns=feature_names, fill_value=0)

    # Make prediction
    features_for_prediction = new_data_df
    prediction = model.predict(features_for_prediction)
    prediction_label = label_encoder.inverse_transform(prediction)
    return prediction_label[0]



# Streamlit app
st.set_page_config(layout='wide')
st.sidebar.image("VSoft-Logo.png")
st.sidebar.header("Insurance Fraud Detection App")
st.sidebar.markdown("The Insurance Fraud Detection System is a cutting-edge application designed to identify and mitigate fraudulent activities within the insurance industry. Utilizing advanced AI algorithms and data analysis techniques, this system analyzes various factors and patterns to detect potential fraud in real-time. By examining user data, transaction details, claim histories, and behavioral patterns, the system provides a robust mechanism to safeguard insurance companies from financial losses due to fraudulent claims. The Insurance Fraud Detection System is an essential tool for modern insurance companies, helping them stay ahead of fraudsters and protect their financial integrity. By leveraging the power of AI and machine learning, this system provides a proactive approach to fraud detection, ensuring a safer and more reliable insurance ecosystem.")

# Left, center, and right columns
st.subheader('Enter any field to predict fraud detection for the customer.')
col1, col3, col5 = st.columns(3)

with col1:
    policy_number = st.text_input('Policy Number')
with col3:
    mobile_number = st.text_input('Mobile Number')
with col5:
    email_id = st.text_input('Email Id')

inputData = None
columnName = None

# Predict button
recommended_button = st.button('Predict Fraud', type='primary')

def getDataFromCSV(columnName, inputData):
    with open("fraud_testdata.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[columnName] == inputData:
                return row    

if recommended_button:
    if policy_number:
        inputData = policy_number
        columnName = 'Policy Number'
    elif mobile_number:
        inputData = mobile_number
        columnName = 'Mobile Number'
    elif email_id:
        inputData = email_id
        columnName = 'Email Id'
    else:
        inputData = None
        columnName = None
    
    if inputData:
        with st.spinner(text="In progress..."):
            row = getDataFromCSV(columnName, inputData)
            print(f'CustomerData: {row}')
            fraud_prediction = predict_fraud(row)
            
            with st.container():
                new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{fraud_prediction}</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                st.write('Fraud detection is crucial in protecting the financial integrity of insurance companies. Early detection of fraudulent activities helps in preventing significant losses.')
            
            customer_details = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">Customer Details:</p>'
            st.markdown(customer_details, unsafe_allow_html=True)
            dataframe = pd.DataFrame.from_dict(row, orient='index')
            st.write(dataframe.T)
    else:
        st.error("Error: No input provided. Please enter data in at least one input field.")
