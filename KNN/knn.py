import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score


# Load dataset
# data = pd.read_excel("C:/SAJIL/KNN/food_knn_dataset-2.xlsx")
data = pd.read_excel(r"C:\SAJIL\KNN\food_knn_dataset-2.xlsx")
print(data)


# Features and target
X = data[['Feature1', 'Feature2']]
y = data['Feature3']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)


# Train model
knn.fit(X_train, y_train)


# Predict test data
y_pred = knn.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 

precision = precision_score(
    y_test, y_pred, average='macro', zero_division=1
)
print("Precision:", precision)

f1 = f1_score(
    y_test, y_pred, average='macro', zero_division=1
)
print("F1:", f1)


# Predict a new food sample
new_sample = pd.DataFrame(
    [[6, 4]],
    columns=['Feature1', 'Feature2']
)

new_prediction = knn.predict(new_sample)

print("The food is:", new_prediction[0])