# Artificial-Intelligence-techniques
Applied AI assignment exploring customer churn prediction, flood risk classification, and news headline detection using logistic regression, neural networks, and NLP with TensorFlow and scikit-learn.

# Artificial Intelligence Techniques — AIA Assignment

A collection of machine learning and deep learning notebooks covering three applied AI problems: customer churn prediction, flood risk classification, and news headline classification.

---

## 📁 Project Structure

```
├── Question 2 FINAL.ipynb   # Customer Churn Prediction (Logistic Regression)
├── Question 3 FINAL.ipynb   # Flood Risk Prediction (Neural Network)
├── Question 4 FINAL.ipynb   # News Headline Classification (NLP + Neural Network)
├── customer_dataset.csv     # Dataset for Q2
├── flood_dataset.csv        # Dataset for Q3
└── article_headlines.csv    # Dataset for Q4
```

---

## 📌 Questions Overview

### Question 2 — Customer Churn Prediction
**Technique:** Logistic Regression  
**Dataset:** `customer_dataset.csv`

Predicts whether a customer will churn using a logistic regression classifier with balanced class weights. Includes:
- Train/test split with stratification
- Feature scaling with `StandardScaler`
- Evaluation via accuracy, precision, and classification report
- Discussion on handling class imbalance (SMOTE, class weights)
- Comparison with alternative models (Random Forest, XGBoost)

---

### Question 3 — Flood Risk Prediction
**Technique:** Deep Neural Network (Keras/TensorFlow)  
**Dataset:** `flood_dataset.csv`

Binary classification model predicting flood risk using a feedforward neural network. Includes:
- Dense + Dropout layers to prevent overfitting
- Early stopping callback
- Precision tracked across epochs
- Analysis of training vs. test generalization gap

---

### Question 4 — News Headline Classification
**Technique:** NLP + Embedding Neural Network (Keras/TensorFlow)  
**Dataset:** `article_headlines.csv`

Classifies news headlines using a text-based deep learning pipeline. Includes:
- Keras `Tokenizer` and sequence padding
- Embedding + GlobalAveragePooling + Dense architecture
- Evaluated on accuracy and precision

---

## 🛠️ Tech Stack

- Python 3
- scikit-learn
- TensorFlow / Keras
- pandas, NumPy

## ▶️ Getting Started

```bash
pip install pandas numpy scikit-learn tensorflow
jupyter notebook
```

Open any of the `.ipynb` files and run the cells in order. Make sure the corresponding CSV dataset is in the same directory as the notebook.
