import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset with the correct delimiter
df = pd.read_csv(r'C:\Users\Juhi Roshan\OneDrive\Desktop\VI\Prodigy - DS\Task 1 - Sale\bank-full.csv')

# Display the first few rows and information about the dataset
print(df.head())
print(df.info())

# Display the column names to verify
print(df.columns)

# Separate the target variable from the features
Y = df['y']
X = df.drop('y', axis=1)

# Convert categorical variables to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Perform a correlation analysis to identify significant features
corr_matrix = df.corr()
print("Correlation with Target Variable:\n", corr_matrix['y'].sort_values(ascending=False))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Build and train the Decision Tree classifier with entropy criterion
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model using cross-validation
cross_val_scores = cross_val_score(clf, X, Y, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(cross_val_scores):.2f}")

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Display the feature importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
print("Feature Importances:\n", feature_importances.sort_values(ascending=False))

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

# Save the plot if needed
plt.savefig('decision_tree.png')
