import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import pickle


# Load dataset
data = pd.read_csv('creditcard_2023.csv')

# Check the first few rows of the dataset
print(data.head())

# Separate features and target
x = data.drop(['id','V2','V3','V4','V5','V6','V7','V8','V9', 'Class'], axis=1)
y = data['Class']

# Normalize the Amount feature



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=5)


# Initialize the decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


