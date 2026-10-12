
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load dataset
df = pd.read_csv("house_prices.csv")

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Select input features and target
X = df[["Area_sqft", "Bedrooms", "Age_years"]]
y = df["Price_lakhs"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE:", round(mae, 2), "lakhs")
print("RMSE:", round(rmse, 2), "lakhs")
print("R2 Score:", round(r2, 3))

# 7. Display regression coefficients
print("\nModel Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 3))

print("Intercept:", round(model.intercept_, 3))

# 8. Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 9. Predict the price of a new house
new_house = pd.DataFrame({
    "Area_sqft": [1500],
    "Bedrooms": [3],
    "Age_years": [5]
})

predicted_price = model.predict(new_house)[0]

print(
    "\nEstimated house price:",
    round(predicted_price, 2),
    "lakhs"
)