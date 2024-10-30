# neural.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import os

# Load and preprocess the data
data = pd.read_csv('data.csv', encoding='utf-8')

# Remove outliers from the Price column
Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5 * IQR
high = Q3 + 1.5 * IQR
data = data[(data['Price'] >= low) & (data['Price'] <= high)]

# One-hot encode the 'Type' column
data = pd.get_dummies(data, columns=['Type'], drop_first=True)

# Split features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Standardize features and target
scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y.values.reshape(-1, 1))

# Model definition
model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(1, activation='linear')
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Function to train and save the model weights
def train_and_save_model():
    print("Training the model...")
    model.fit(scaler_X.transform(X), scaler_y.transform(y.values.reshape(-1, 1)), epochs=200, batch_size=32, validation_split=0.2)
    model.save_weights("model_weights.weights.h5")
    print("Model trained and weights saved.")

# Function to load weights if available, otherwise train and save
def load_or_train_model():
    if os.path.exists("model_weights.weights.h5"):
        model.load_weights("model_weights.weights.h5")
        print("Model weights loaded.")
    else:
        train_and_save_model()

# Ensure model is loaded or trained on import
load_or_train_model()

# Prediction function
def predict_price(rooms, house_type, postcode, distance):
    # Prepare input data
    input_data = pd.DataFrame([[rooms, postcode, distance]], columns=['Rooms', 'Postcode', 'Distance'])
    
    # One-hot encoding for the 'Type' field
    if house_type == 't':  # Townhouse
        input_data['Type_t'] = 1
        input_data['Type_u'] = 0
    elif house_type == 'u':  # Unit
        input_data['Type_t'] = 0
        input_data['Type_u'] = 1
    else:  # Default to house
        input_data['Type_t'] = 0
        input_data['Type_u'] = 0

    # Ensure input data has the same structure as training data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]  # Reorder columns for consistency

    # Scale input data
    input_data_scaled = scaler_X.transform(input_data)
    
    # Make a prediction
    scaled_prediction = model.predict(input_data_scaled)
    
    # Convert the scaled prediction back to original price scale
    predicted_price = scaler_y.inverse_transform(scaled_prediction)[0][0]
    
    return predicted_price
