"""
# source: chatGpt
XGBoost, short for eXtreme Gradient Boosting, is a powerful machine learning algorithm that belongs to the ensemble learning category. Here's a simplified explanation of how the XGBoost algorithm works:

1. **Boosting:**
   - XGBoost is based on the concept of boosting, where weak learners (typically shallow decision trees) are sequentially added to form a strong learner.
   - Each tree corrects the errors of the previous ones, with more weight given to instances that were misclassified.

2. **Gradient Boosting:**
   - XGBoost extends boosting by incorporating a gradient descent optimization technique.
   - It minimizes a specific loss function by adjusting the parameters of the weak learners, making the model more accurate with each iteration.

3. **Decision Trees:**
   - XGBoost builds decision trees to make predictions.
   - Each tree is constructed by selecting the best split points for the features, aiming to reduce the overall prediction error.

4. **Regularization:**
   - XGBoost includes regularization terms in its objective function to control the complexity of the individual trees and prevent overfitting.
   - Regularization helps ensure the model generalizes well to new, unseen data.

5. **Feature Importance:**
   - XGBoost provides insights into feature importance.
   - It assigns weights to features based on their contribution to reducing the loss function during tree construction, allowing users to understand which features are crucial for predictions.

6. **Parallel Processing:**
   - XGBoost is designed for efficiency and speed.
   - It leverages parallel processing capabilities to build trees in parallel, making it computationally efficient and scalable.

7. **Cross-validation:**
   - XGBoost supports k-fold cross-validation during training, helping to evaluate the model's performance more reliably and mitigate overfitting.

8. **Handling Missing Values:**
   - XGBoost can handle missing values in the dataset during training, eliminating the need for imputation or removal of incomplete data.

* summary: XGBoost combines the principles of boosting, gradient descent, and regularization.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Sample dataset for binary classification
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'target': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['feature1', 'feature2']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost classifier
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
