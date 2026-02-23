#https://colab.research.google.com/drive/1H9vHGxzn-Xpwd6CbCbyCU7JUNFw-e-Yf?usp=sharing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

data = pd.read_csv(f"C:\\Users\\hardi\\Downloads\\creditcard.csv")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display basic information about the dataset
print("\nDataset shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nData types:")
print(data.dtypes)
print("\nBasic statistics:")
print(data.describe())



data.duplicated().sum()

# calculate the number of fraud and genuine transactions

# percentage value :fraud

geninue_transaction= data[data['Class'] == 0].shape[0]
fraud_transaction = data[data['Class'] == 1].shape[0]

print("Number of genuine transactions:", geninue_transaction)
print("Number of fraud transactions:", fraud_transaction)

fraud_perceantage = (fraud_transaction / data.shape[0]) * 100

print(f"Percentage of fraud transactions: {fraud_perceantage:.4f}%")

data['Class'].value_counts(normalize=True) * 100

# Visualize the distribution of the target variableq

sns.countplot(x='Class', data=data)
plt.title('Distribution of Class Variable')
plt.xlabel('Class (0: Genuine, 1: Fraud)')
plt.ylabel('Count')
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

data

from imblearn.over_sampling import SMOTE

# Separate features and target variable
from sklearn.model_selection import train_test_split

x = data.drop(columns=['Class', 'Time', 'Amount'])

y = data['Class']

# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


# Train a model by Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(x_train_smote, y_train_smote)

dt_prediction = dt_classifier.predict(x_test)
dt_accuracy = dt_classifier.score(x_test, y_test)
print("Decision Tree Accuracy:", dt_accuracy)

print(classification_report(y_test,dt_prediction))

# model says 10 transaction are fraud

#but actually data  7 are fraud

# 7/10 = 0.70

# high precision = model give accurate fraud answers


# 100% precision has correctly identify all normal transaction

# 40% precision has correctly indentify fraud transaction



# recall : how many real frauds data the model caught

# there are 10 real frauds

# model caught only 6

# 6/10 : 0.60    # high recall : catches more fraud


# f1 score : precision + recall

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
random_forest = RandomForestClassifier()
random_forest.fit(x_train_smote, y_train_smote)

rf_prediction = random_forest.predict(x_test)
rf_accuracy = random_forest.score(x_test, y_test)
print(f"Random Forest Accuracy:", rf_accuracy)


# SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(x_train_smote, y_train_smote)

svm_prediction = svm_classifier.predict(x_test)
svm_accuracy = svm_classifier.score(x_test, y_test)

print(f"SVM Accuracy:", svm_accuracy)