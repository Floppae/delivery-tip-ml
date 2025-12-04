import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
print("\n" + "=" * 10 + " Our Linear Models " + "=" * 10)

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

# How do we know that the parameters we set (alpha=0.1, alpha=1.0, degree=2) are the most optimal?
# We do not. Thus, we can play around with different parameters to see which parameters allow for optimal performance
print("=" * 10 + " Hyperparameter Tuning " + "=" * 10)

# Test alphas for Ridge and Lasso Regressions
print("\nTesting Ridge and Lasso Regression alphas:")
alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0]
best_ridge_alpha = None
best_ridge_mae = float('inf')
best_lasso_alpha = None
best_lasso_mae = float('inf')

for alpha in alphas_to_test:

    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
    
    if ridge_mae < best_ridge_mae:
        best_ridge_mae = ridge_mae
        best_ridge_alpha = alpha

    print(f"RIDGE alpha = {alpha} | MAE: {ridge_mae:.3f}")

    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

    if lasso_mae < best_lasso_mae:
        best_lasso_mae = lasso_mae
        best_lasso_alpha = alpha
    
    print(f"LASSO alpha = {alpha} | MAE: {ridge_mae:.3f}")

print(f"\nBest Ridge alpha: {best_ridge_alpha} (MAE: {best_ridge_mae:.3f})")
print(f"Best Lasso alpha: {best_lasso_alpha} (MAE: {best_lasso_mae:.3f})")

# Test degrees for polynomial features
print("\nTesting Polynomial Regression degrees:")
test_degrees = [2, 3, 4]
best_poly_degree = None
best_poly_mae = float('inf')

for degree in test_degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    poly_mae = mean_absolute_error(y_test, y_pred_poly)
    
    print(f"degree={degree} | MAE: {poly_mae:.3f}")
    
    if poly_mae < best_poly_mae:
        best_poly_mae = poly_mae
        best_poly_degree = degree

print(f"Best Polynomial degree: {best_poly_degree}\n")

# To capture nonlinear trends, the code below utilizes polynomial features to capture trends along curves instead of along straight lines
# With polynomial features, we simply create more features with the feature raised to "degree"
# Essentially, we create more features for the model to work with, adding flexibility resulting in the ability to capture curves
# It's important to not make the degree too high as this would cause overfitting and unecessary computational cost
print("=" * 10 + " Polynomial Regression " + "=" * 10)

# Create polynomial features for model + fit the features
# Additionally, create a test set to test out model with
poly = PolynomialFeatures(degree=best_poly_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Print amount of columns before and after adding polynomial features (.shape returns (row,col) number so .shape[1] returns number of columns, or, features)
print(f"Starting number of features: {X_train_scaled.shape[1]}")
print(f"Resulting number of features: {X_train_poly.shape[1]}")

# Once we have our new features, fit a model with the new features, create prediction set, calculate error based on actual data.
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred = poly_model.predict(X_test_poly)
mae_poly = mean_absolute_error(y_test, y_pred)

print(f"Polynomial Regression MAE: {mae_poly:.3f}")
if(mae_poly < best_mae):
    print(f"Polynomial Regression performed better than linear Regression by {best_mae - mae_poly:.3f}")
else:
    print(f"Linear Regression performed better than Polynomial Regression by {mae_poly - best_mae:.3f}")