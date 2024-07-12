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
import faker
from datetime import datetime, time
from streamlit_card import card
# Create a sample dataset
# Initialize Faker
fake = faker.Faker()

# Generate random customer data
# def generate_customer_data(num_records):
#     data = []
#     for _ in range(num_records):
#         policy_number = fake.random_number(digits=5, fix_len=True)
#         customer_id = fake.random_number(digits=3, fix_len=True)
#         customer_name = fake.name()
#         mobile_number = fake.phone_number()
#         email_id = fake.email()
#         age = random.randint(18, 70)
#         gender = random.choice(['Male', 'Female'])
#         marital_status = random.choice(['Married', 'Single'])
#         dependents = random.randint(0, 5)
#         location = fake.city()
#         occupation = fake.job()
#         claims_3yrs = random.randint(0, 3)
#         claim_type = random.choice(['Accident', 'Theft', 'Natural Disaster'])
#         online_activity = random.randint(0, 100)
#         service_calls = random.randint(0, 10)
#         current_insurance = random.choice(['Home Insurance A', 'Auto Insurance B', 'Health Insurance C', 'Life Insurance D'])
#         coverage = random.choice(['Full', 'Partial'])
#         renewal_date = fake.date_between(start_date='+1d', end_date='+2y').strftime('%Y-%m-%d')
#         premium_3yrs = random.randint(1000, 5000)
#         # recommended_insurance = random.choice(['Health Insurance', 'Life Insurance', 'Accident Insurance', 'Vision Insurance','Dental Insurance','Disability Insurance','Critical illness Insurance','Hospital indemnity insurance'])
#         page_visits_health = random.randint(0, 20)
#         page_visits_auto = random.randint(0, 20)
#         page_visits_home = random.randint(0, 20)
#         page_visits_life = random.randint(0, 20)
#         time_spent_health = random.randint(0, 120)
#         time_spent_auto = random.randint(0, 120)
        
#         data.append([
#             policy_number, customer_id, customer_name, mobile_number, email_id, age, gender,
#             marital_status, dependents, location, occupation, claims_3yrs, claim_type, 
#             online_activity, service_calls, current_insurance, coverage, renewal_date,
#             premium_3yrs, page_visits_health, page_visits_auto,
#             page_visits_home, page_visits_life, time_spent_health, time_spent_auto
#         ])
    
#     return data

# # Define columns
# columns = [
#     'Policy Number', 'Customer ID', 'Customer Name', 'Mobile Number', 'Email Id', 'Age', 'Gender',
#     'Marital Status', 'Dependents', 'Location', 'Occupation', 'Claims (3yrs)', 'Claim Type',
#     'Online Activity', 'Service Calls', 'Current Insurance', 'Coverage', 'Renewal', 'Premium (3yrs)',
#     'Page_Visits_Health', 'Page_Visits_Auto', 'Page_Visits_Home', 
#     'Page_Visits_Life', 'Time_Spent_Health', 'Time_Spent_Auto'
# ]

# # Generate data for 500 customers
# customer_data = generate_customer_data(30)

# # Create DataFrame
# df = pd.DataFrame(customer_data, columns=columns)

# # Save DataFrame to CSV
# df.to_csv('test_data.csv', index=False)

# print('customer_data.csv file created successfully with 500 records')


# Load the dataset
data = pd.read_csv('customer_data.csv')

# Convert 'Renewal' date to the number of days from today
today = pd.Timestamp.today()
data['Renewal'] = pd.to_datetime(data['Renewal'])
data['Days_Until_Renewal'] = (data['Renewal'] - today).dt.days
data = data.drop(columns=['Renewal'])

# Encode categorical variables
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Encode 'Marital Status' using OneHotEncoder
encoder_marital_status = OneHotEncoder(sparse_output=False)
marital_status_encoded = encoder_marital_status.fit_transform(data[['Marital Status']])
marital_status_encoded_df = pd.DataFrame(marital_status_encoded, columns=encoder_marital_status.get_feature_names_out())

# Encode 'Claim Type' using OneHotEncoder
encoder_claim_type = OneHotEncoder(sparse_output=False)
claim_type_encoded = encoder_claim_type.fit_transform(data[['Claim Type']])
claim_type_encoded_df = pd.DataFrame(claim_type_encoded, columns=encoder_claim_type.get_feature_names_out())

# Encode 'Coverage' using OneHotEncoder
encoder_coverage = OneHotEncoder(sparse_output=False)
coverage_encoded = encoder_coverage.fit_transform(data[['Coverage']])
coverage_encoded_df = pd.DataFrame(coverage_encoded, columns=encoder_coverage.get_feature_names_out())

# Combine the original data with the encoded columns and drop unnecessary columns
data = pd.concat([data, marital_status_encoded_df, claim_type_encoded_df, coverage_encoded_df], axis=1).drop(columns=['Policy Number', 'Customer ID', 'Customer Name', 'Mobile Number', 'Email Id', 'Location', 'Occupation', 'Current Insurance', 'Claim Type', 'Marital Status', 'Coverage'])

# Define features and target
X = data.drop(columns=['Recommended Insurance'])
y = data['Recommended Insurance']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=21)
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

def predict_recommended_insurance(new_data_dict):
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
      'Premium (3yrs)': [new_data_dict['Premium (3yrs)']],
      'Page_Visits_Health': [new_data_dict['Page_Visits_Health']],
      'Page_Visits_Auto': [new_data_dict['Page_Visits_Auto']],
      'Page_Visits_Home': [new_data_dict['Page_Visits_Home']],
      'Page_Visits_Life': [new_data_dict['Page_Visits_Life']],
      'Time_Spent_Health': [new_data_dict['Time_Spent_Health']],
      'Time_Spent_Auto': [new_data_dict['Time_Spent_Auto']],
      'Days_Until_Renewal': [(pd.Timestamp('2024-01-01') - pd.Timestamp.today()).days]
    }

    # Convert dictionary to DataFrame
    new_data_df = pd.DataFrame(new_data_dict)

    # Preprocess the new data similarly
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
# Define your custom HTML for the background
st.markdown("""<style>.stApp {background-image: url('/static/bg.png');background-size: cover;}</style>""", unsafe_allow_html=True)
st.sidebar.image("VSoft-Logo.png")
st.sidebar.header("Insurance Recommendation App")
st.sidebar.markdown("The Insurance Recommendation App is designed to provide personalized insurance suggestions based on user input and data analysis. It utilizes advanced AI algorithms to assess user data, preferences,risk factors and company browsiing history to offer tailored recommendations for various insurance products such as life insurance, health insurance, acciden insurance, vision inscurance and more. The app aims to streamline the insurance selection process, ensuring users receive optimal coverage that meets their individual needs and financial circumstances")

# left_co, cent_co,last_co = st.columns(3)

st.subheader('Enter any field to predict insurance recommendation for the customer.')
# Set up the layout with two columns
col1, col3, col5 = st.columns(3)


with col1:
    policy_number = st.text_input('Policy Number')
with col3:
    
    mobile_number = st.text_input('Mobile Number')

with col5:
    
    email_id = st.text_input('Email Id      ')

inputData = None
columnName = None

# Predict button
# if st.button('Predict Recommended Insurance'):         
   

st.markdown('')
recommended_button =  st.button(':white[Recommended Insurance]', type='primary')
def  getDataFromCSV(columnName,inputData):
  #with open("insurance_data.csv") as csvfile: 
  with open("test_data.csv") as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
        if (row[columnName] == inputData):
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
    #Handle the inputData    
    if inputData:
        st.spinner(text="In progress...")
        row = getDataFromCSV(columnName,inputData)
        print(f'CustomerData:{row}')
        recommended_insurance_text = predict_recommended_insurance(row)
        
        with st.container(border=True):
            new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{recommended_insurance_text}</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            # st.subheader(f'{recommended_insurance_text}')
            st.write('What is life insurance? First and foremost, it’s a way to protect your family and those who depend on you for financial support. It can provide a large, income tax-free payout to help them carry on if you pass away unexpectedly — and some policies have features that can help build family assets')
        customer_details = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">Customer Details:</p>'
        st.markdown(customer_details, unsafe_allow_html=True)
        dataframe=pd.DataFrame.from_dict(row,orient='index')
            # Display the dataframe with enhanced UI
        #dfR = pd.DataFrame(row)       
        st.write(dataframe.T)
    
    else:
        st.error("Error: No input provided. Please enter data in at least one input field.") 
    

    
        
    
         
            

          


   