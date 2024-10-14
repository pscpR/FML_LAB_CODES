import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Plot feature importances
feature_importance = model.feature_importances_
plt.barh(data.feature_names, feature_importance, color='teal')
plt.xlabel('Feature Importance')
plt.title('Decision Tree - Feature Importance')
plt.show()
