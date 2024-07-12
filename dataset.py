import faker
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
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

# Initialize Faker
fake = faker.Faker()

# Generate random customer data
def generate_customer_data(num_records):
    data = []
    for _ in range(num_records):
         policy_number = fake.random_number(digits=5, fix_len=True)
        customer_id = fake.random_number(digits=3, fix_len=True)
        customer_name = fake.name()
        mobile_number = fake.phone_number()
        email_id = fake.email()
        age = random.randint(18, 70)
        gender = random.choice(['Male', 'Female'])
        marital_status = random.choice(['Married', 'Single'])
        dependents = random.randint(0, 5)
        location = fake.city()
        occupation = fake.job()
        claims_3yrs = random.randint(0, 3)
        claim_type = random.choice(['Accident', 'Theft', 'Natural Disaster'])
        online_activity = random.randint(0, 100)
        service_calls = random.randint(0, 5)
        current_insurance = random.choice(['Home Insurance A', 'Auto Insurance B', 'Health Insurance C', 'Life Insurance D'])
        coverage = random.choice(['Full', 'Partial'])
        premium_3yrs = random.randint(1000, 5000)
        fraud = random.choice(['1', '0'])

        data.append([
            policy_number, customer_id, customer_name, mobile_number, email_id, age, gender,
            marital_status, dependents, location, occupation, claims_3yrs, claim_type, 
            online_activity, service_calls, current_insurance, coverage, premium_3yrs, fraud
        ])

    return data

# Define columns
columns = [
    'Policy Number', 'Customer ID', 'Customer Name', 'Mobile Number', 'Email Id', 'Age', 'Gender',
    'Marital Status', 'Dependents', 'Location', 'Occupation', 'Claims (3yrs)', 'Claim Type', 
    'Online Activity', 'Service Calls', 'Current Insurance', 'Coverage', 'Premium (3yrs)' , 'Fraud'
]

# Generate random customer data
num_records = 10000
data = generate_customer_data(num_records)

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
df.to_csv('fraud_data.csv', index=False)

print('fraud_data.csv file created successfully with', num_records, 'records')

