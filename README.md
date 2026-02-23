# 💳 Credit Card Fraud Detection System

A Machine Learning project focused on detecting fraudulent credit card transactions using classification algorithms and handling class imbalance with SMOTE.

---

## 🚀 Project Overview

This project aims to build a fraud detection system capable of identifying fraudulent credit card transactions from highly imbalanced financial data.

The project follows a structured ML pipeline including:

- Data Exploration
- Data Preprocessing
- Class Imbalance Handling (SMOTE)
- Model Training
- Model Evaluation & Comparison

---

## 🛠 Tech Stack & Libraries

* **Language:** Python  
* **Data Analysis:** Pandas, NumPy  
* **Visualization:** Seaborn, Matplotlib  
* **Machine Learning:**  
  * Decision Tree  
  * Random Forest  
  * Support Vector Machine (SVM)  
* **Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique)  
* **Model Evaluation:** Accuracy, Precision, Recall, F1-Score  

---

## 📊 Key Features & Methodology

To ensure reliable fraud detection performance, the following techniques were implemented:

* **Data Exploration & Analysis:**  
  - Checked dataset structure, duplicates, and statistics  
  - Analyzed fraud vs genuine transaction distribution  
  - Visualized class imbalance using count plots  

* **Data Preprocessing:**  
  - Normalized transaction amount using StandardScaler  
  - Removed unnecessary columns (Time, Amount)  
  - Separated features and target variable  

* **Class Imbalance Handling:**  
  - Applied **SMOTE** to balance fraud and genuine classes  
  - Improved model ability to detect minority fraud cases  

* **Model Training & Comparison:**  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - Support Vector Machine (SVM)  
  - Compared model performance across metrics  

* **Model Evaluation:**  
  - Accuracy Score  
  - Precision (Fraud Detection Reliability)  
  - Recall (Fraud Catch Rate)  
  - F1 Score (Balanced Performance Metric)  

---

## 📈 Model Performance

The models were evaluated using classification metrics:

- Precision → How accurately fraud cases are predicted  
- Recall → How many real frauds are detected  
- F1 Score → Balance between precision and recall  
- Accuracy → Overall correctness  

Random Forest showed strong performance due to ensemble learning advantages.

---

## 📁 Dataset

- Credit Card Transactions Dataset  
- Contains highly imbalanced fraud vs genuine transactions  
- Target Variable: `Class`  
  - 0 → Genuine  
  - 1 → Fraud  

---

## ▶ How to Run the Project

```bash
pip install -r requirements.txt
python Credit Card Fraud Detection.py
