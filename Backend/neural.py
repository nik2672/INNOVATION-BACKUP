import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ensure correct encoding we were having issues hence encoding utf-8 
sys.stdout.reconfigure(encoding='utf-8')

# we were also havign isseus with loading the data to added in error message if it cannot read the file

try:
    data = pd.read_csv('data.csv', encoding='utf-8')
    print("Data is loaded")
except Exception as e:
    print(f"file is not being read {e}")
    sys.exit(1)

# we had issues reading price collumn this will check that the price column does in fact exist 
if 'Price' not in data.columns:
    raise KeyError(" the 'Price' column not found")

# get rid of any outliers in the dataset before processing it into the model 
Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5 * IQR  
high = Q3 + 1.5 * IQR  

# save the initial structure of the dataset similar to the other models so we can print out how many lines are gone when removing outliers 
initial_dimensions = data.shape  

# get rid of any outliers and print how many have been removed 
data = data[(data['Price'] >= low) & (data['Price'] <= high)]  
print(f"data dimentions after the outliers are removed: {data.shape}")
print(f"data points that were removed: {initial_dimensions[0] - data.shape[0]}")

# similar to modelClass we encoded our post code collumn to be replaced by postcode mean price in order to leverage average property prices associated with postcode, this improved the performacne 
# calcualte the mean
postcode_mean_price = data.groupby('Postcode')['Price'].mean()
# map built in function will then replace each postcode in the 'postcode' collumn with its mean price
data['Postcode_encoded'] = data['Postcode'].map(postcode_mean_price)

# axis =1 refers to column we drop it as we replaced it with mean price per postcode 
data = data.drop('Postcode', axis=1)

# set target variables 
X = data.drop('Price', axis=1)
y = data['Price']

# more encoding/feature engineering identifying the 'type' feature 
categorical_features = ['Type']

# again we had issues wiht identifying certian features this is to check if it exists, this for loop will iterate over the dataframe 
for feature in categorical_features:
    if feature not in X.columns:
        raise KeyError(f"The '{feature}' hasnt been foound.")

# convert the features (categorial) to string ensurng the data is in correct format
X[categorical_features] = X[categorical_features].astype(str)

# we convert the catergory to dummy variables in for isntance 'house' will be 'Type_House' we aimed to improve perfromance as the models can understand numerical input better 
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

#will show the structure of the data after ecoding/feature enginerring incase any data is lost and to identify any hidden issues after encoding
print(f"Shape of X after encoding: {X.shape}")

# data is split into training and test , test with 20% of sample 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#prints how many datapoint used for training and testing ensurign the models is handling the data without any errors.
print(f"size of trianing data: {X_train.shape}")
print(f"size of test data: {X_test.shape}")

# similar to modelClassF we attempt to stadardize/normlaise our data in effor to improve how our dataset is read
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# we standardize/normalise the target varable
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten() #flattening helped prevent an error we faced by converting the array to 1D array 
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# we increaseed the depth of our neural network in order to get better r^2 adding three layers compareed to original
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
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

# to improve performane we attempt to use Adaptive Moment Esitmation (adam) in order to adapt the learning rate, we also set learning rate to 0.0001 as slower rate may result in more precise results
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# using tensorflow/keras lirbary we use inbuiilt function to summarise our model architecture.
print("\nModel Summary:")
model.summary()

# train model, we also increased number of epochs from 100 to 200 yielding imrpoved scores
history = model.fit(
    X_train, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2, #to validate perfromance we split it 20%
    verbose=1
)

# create loss over epochs graph 
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red') #changed to red more visible
plt.title('Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right') #create a legend to improve performacne 
plt.grid(True)
plt.show()

# returns predicted values which were 'scaled ' 
y_pred_scaled = model.predict(X_test)

# we inverse / cnvert the scaled values we used to training the model back to normal to compare them with 'actual prices' 
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten() #added flatten incase we ran into any errors 

# metics mse and rmse calcaulted
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # added rmse using numpy lirbary 
print(f'\nMean Squared Error (MSE) : {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# calaulte r squared, MAE, and also compared mean price from the test with the mean price that was predicted by the model 
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mean_price = y_test.mean()
predicted_mean_price = y_pred.mean()

# print all the metric results
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Actual Price: {mean_price}')
print(f'Mean Predicted Price: {predicted_mean_price}')


# visulise the predicted and actual prices in a scatter plot 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') #mkae the lines red 
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted comparison with Actual Prices')
plt.grid(True)
plt.show()
