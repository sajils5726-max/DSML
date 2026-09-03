from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes(as_frame=True)

df = diabetes.frame

print("Dataset Loaded Successfully!\n")

print(df.head())

X = df[['bmi', 'bp', 's1', 's5']]

y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

print("\n Regression Equation:")

print(f"y = {model.intercept_:.2f}", end="")

for feature, coef in zip(X.columns, model.coef_):
    print(f" + ({coef:.2f} × {feature})", end="")

print("\n")

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(" Model Performance Metrics:")

print("Mean Squared Error (MSE):", round(mse, 2))

print("R² Score:", round(r2, 2))

new_data = pd.DataFrame([[0.05, 0.03, 0.02, 0.04]], columns=['bmi', 'bp', 's1', 's5'])

predicted_value = model.predict(new_data)

print("\n Predicted Target Value for New Data:", round(predicted_value[0], 2))

plt.figure(figsize=(6,5))

plt.scatter(y_test, y_pred, color='blue', s=60)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted (Multiple Linear Regression)")

plt.grid(True)

plt.show()