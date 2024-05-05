# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, precision_recall_curve
#load the dataset
data = pd.read_csv('Churn_Modelling.csv')
print("CUSTOMER CHURN PREDICTION USING RANDOM FOREST CLASSIFIER")
print(data.columns)
# preprocessing 
data.dropna(inplace=True)
data_encoded = pd.get_dummies(data, columns=['Geography', 'Gender'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_encoded[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']])
data_encoded[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumofProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']] = scaled_features
plt.figure(figsize=(8, 6))
sns.countplot(x='Exited', data=data_encoded)
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()
# VISULISATION
numeric_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data_encoded[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# random forest classifier
X = data_encoded.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
y = data_encoded['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# accuracy report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# display ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
# precision curve
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
