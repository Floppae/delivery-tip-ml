import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error

# Load synthetic data we previously generated
# Note: change filepath based on current directory
df = pd.read_csv("../data/synthetic/synthetic_delivery_data.csv")

# Distinguish the features we will predict tip with from the calculated tip
X = df[['distance_miles', 'order_subtotal', 'wait_time_minutes', 'communication_rating', 'item_count', 'messages_sent']]
y = df['tip_percent']

# Split the features and calculated tip into training and testing data. (80% training 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" +"=" * 10 + " Data Overview " + "=" * 10)
print(f"Training set: {len(X_train)} entries")
print(f"Testing set: {len(X_test)} entries")

# We must standardize the features because the model will punish features in lasso and ridge regressiosn too harshly
# For linear regressions, standardizing isn't necessary because the main goal is to minimize prediction error
# However, Lasso and Ridge regressions aim to minimize prediction error AND penalize large coefficients
# So if we don't standardize each feature to have mean=0 sd=1, lasso and ridge regressions will unfairly punish features with low ranges
# For example, order_total=50, rating=5, calculates some tip where the coefficient for rating must be far greater than the coefficient for order_total because otherwise rating would have too insignificant of an impact
# Ridge and Lasso Regressions would see this large coefficient for rating and unfairly punish it for being too far from 0.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a dictionary (key: name of model | value: model) to hold our models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

# We will use 3 models: Linear, Lasso, Ridge and see which one is best
print("\n" + "=" * 10 + " Our Models " + "=" * 10)

# Iterate through each model, fit it, predict values with it, and calculate MAE.
# Save the values into a second dictionary (key:name | value:object containing model performance)
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'mae': mae
    }
    
    print(f"{name} MAE: {mae:.3f}")

# Initialize best_model with the first model in results
# Iterate over each model to find the model with the lowest MAE
first_model_name = list(results.keys())[0]
best_model = [first_model_name, results[first_model_name]]
for name, model in results.items():
    if model['mae'] < best_model[1].get('mae', 0):
        best_model = [name, model]

# Grab our best model name and mae and print it out
best_model_name = best_model[0]
best_mae = best_model[1].get('mae', 0)
print("\n" + "=" * 10 + " Best Model " + "=" * 10)
print(f"{best_model_name} MAE: {best_mae:.3f}\n")