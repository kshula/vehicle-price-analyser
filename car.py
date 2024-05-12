import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("car_details.csv")

# Drop rows with missing values in relevant columns
df = df.dropna(subset=['Price', 'Kilometer', 'Seating Capacity', 'Fuel Tank Capacity'])

# Prepare data for modeling
X = df[['Kilometer', 'Seating Capacity', 'Fuel Tank Capacity']]
y = df['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Gradient Boosting model
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Predict on test set
y_pred = gb_model.predict(X_test)

# Calculate R^2 score and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print model performance metrics
print(f"Gradient Boosting Model Performance:")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")

# Extract feature names
feature_names = list(X.columns)

# Add constant term to the independent variables (X) for the intercept in regression
X_with_const = sm.add_constant(X)

# Fit ordinary least squares (OLS) regression model
model = sm.OLS(y, X_with_const).fit()

# Get coefficients and p-values for Gradient Boosting model
gb_coefficients = gb_model.feature_importances_
gb_pvalues = [model.pvalues[feature] for feature in ['Kilometer', 'Seating Capacity', 'Fuel Tank Capacity']]

# Print Gradient Boosting model coefficients and significance
print("\nGradient Boosting Model Coefficients:")
for i in range(len(feature_names)):
    print(f"{feature_names[i]} Coefficient: {gb_coefficients[i]:.4f} (p-value: {gb_pvalues[i]:.4f})")

# Hypothesis testing for significance at alpha = 0.05
alpha = 0.05
for i in range(len(feature_names)):
    if gb_pvalues[i] < alpha:
        print(f"{feature_names[i]} coefficient is statistically significant.")
    else:
        print(f"{feature_names[i]} coefficient is not statistically significant.")
