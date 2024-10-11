import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Create 'dataset' directory if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Define the path to the dataset file
dataset_file = 'dataset/electricity.csv'

# Check if the dataset exists locally, if not, download it
if not os.path.exists(dataset_file):
    url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv'
    data = pd.read_csv(url, low_memory=False)
    # Save the dataset locally
    data.to_csv(dataset_file, index=False)
else:
    # Load the dataset from the local file
    data = pd.read_csv(dataset_file, low_memory=False)

# Inspect the columns in the dataset to identify the correct names
print("Dataset columns:", data.columns)

# Handle missing or invalid values ('?' in this case)
# Replace '?' with NaN, and then drop or fill NaN values
data.replace('?', pd.NA, inplace=True)

# Option 1: Drop rows with missing values
data.dropna(inplace=True)

# Option 2: Alternatively, you can fill missing values with the mean or median:
# data.fillna(data.mean(), inplace=True)

# Ensure the columns are in the correct numeric format
data['SystemLoadEA'] = pd.to_numeric(data['SystemLoadEA'], errors='coerce')
data['SMPEA'] = pd.to_numeric(data['SMPEA'], errors='coerce')

# Check if 'SystemLoadEA' and 'SMPEA' columns exist after cleaning
if 'SystemLoadEA' not in data.columns or 'SMPEA' not in data.columns:
    raise ValueError("Column names 'SystemLoadEA' or 'SMPEA' not found in the dataset. Please check the actual column names.")

# Prepare the features (X) and target (y)
X = data[['SystemLoadEA']].values  # Feature: System Load
y = data['SMPEA'].values           # Target: Market Price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model as a pickle file
model_path = 'model/electricity_model.pkl'
if not os.path.exists('model'):
    os.makedirs('model')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully in 'model/model.pkl'.")