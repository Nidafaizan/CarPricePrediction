import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load the data
df = pd.read_csv('carprices (1) (1).csv')

# Extract features and target
X = df[['Mileage', 'Age(yrs)']].values
y = df['Sell Price($)'].values

# Initialize MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")
print(
    f"Model Coefficients: Mileage={model.coef_[0]:.4f}, Age={model.coef_[1]:.4f}")
print(f"Intercept: ${model.intercept_:.2f}")

# Save the model and scaler
with open('car_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved successfully!")
