# QUESTION 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#### Load dataset
df = pd.read_csv("customer_dataset.csv")
print("Dataset loaded. Shape:", df.shape)
print("Churn rate:", "{:.2%}".format(df['Churn'].mean()))

#### Features and target
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

#### Split into training and testing sets with stratification (better for imbalanced classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train set size:", X_train.shape[0], "samples")
print("Test set size:", X_test.shape[0], "samples")

#### Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

#### Show dataset info
df.info()

#### Display first few rows
df.head()
from sklearn.linear_model import LogisticRegression

#### Build logistic regression model with balanced class weights and sufficient iterations
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

#### Train the model on scaled training data
model.fit(X_train_scaled, y_train)
#### Evaluate on test data
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import precision_score
print("Precision:", precision_score(y_test, y_pred))
model
from sklearn.metrics import accuracy_score, precision_score, classification_report

#### Predict the test set results
y_pred = model.predict(X_test_scaled)

#### Calculate accuracy and precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)

#### Print the results
print("Model Evaluation Results:")
print("Accuracy Score:", round(accuracy, 2))
print("Precision Score:", round(precision, 2))

print("\nDetailed Performance Metrics:")
print("Accuracy:", round(accuracy, 4), "(", round(accuracy * 100, 1), "%)")
print("Precision:", round(precision, 4), "(", round(precision * 100, 1), "%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#### Summary metrics
metrics = {
    'Metric': ['Accuracy', 'Precision'],
    'Score': [accuracy, precision]
}
metrics_df = pd.DataFrame(metrics)
metrics_df['Score'] = metrics_df['Score'].round(4)


#### Classification report as dict
report_dict = classification_report(y_test, y_pred, output_dict=True)

#### Convert to DataFrame and round values
report_df = pd.DataFrame(report_dict).transpose().round(4)

print("\nClassification Report:")
report_df

# Question 3 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#### Load the dataset
df = pd.read_csv("flood_dataset.csv")
print("Dataset shape:", df.shape)

#### Separate features and target
X = df.drop(columns=["Flood Risk (1=Yes, 0=No)"])
y = df["Flood Risk (1=Yes, 0=No)"]
print("Features shape:", X.shape, ", Target shape:", y.shape)

#### Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set size:", X_train.shape[0], ", Test set size:", X_test.shape[0])

#### Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#### Define model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),                    # Dropout to prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

#### Compile the model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision'])

#### Early stopping callback to stop training if val_loss doesn't improve for 5 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#### Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)
from sklearn.metrics import precision_score

#### Predict probabilities on test set
y_pred_probs = model.predict(X_test_scaled)

#### Convert probabilities to binary predictions using threshold 0.5
y_pred = (y_pred_probs > 0.5).astype("int32")

#### Calculate precision score on test data
precision = precision_score(y_test, y_pred)

print("Test Precision:", precision)
from sklearn.metrics import precision_score

#### Predict probabilities on the test set
y_pred_probs = model.predict(X_test_scaled)

#### Change classification threshold from 0.5 to 0.8
y_pred = (y_pred_probs > 0.8).astype("int32")

#### Evaluate precision
precision = precision_score(y_test, y_pred)
print("Test Precision:", precision)

#### Show precision after each epoch during training
for i, p in enumerate(history.history['Precision']):
    print("Epoch", i + 1, "Precision:", p)

#### Final precision from training history
final_precision = history.history['Precision'][-1]
print("Final training precision:", final_precision)

# Question 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#### Load data
df = pd.read_csv('article_headlines.csv')
print("Dataset shape:", df.shape)
print("Label distribution:")
print(df['Label'].value_counts())

#### Prepare text and labels
texts = df['Headline'].values
labels = df['Label'].values

#### Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

#### Tokenization
max_features = 10000  # vocabulary size
max_length = 100      # sequence length

tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

#### Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

#### Padding sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

print("Training sequences shape:", X_train_pad.shape)
print("Test sequences shape:", X_test_pad.shape)

df['Label'].value_counts()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D, Embedding, Dense
from tensorflow.keras.models import Sequential

#### Build model
model = Sequential([
    Embedding(max_features, 128, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

#### Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture:")
model.summary()
#### Train model
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#### Make predictions
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

#### Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy:", round(accuracy, 4), "(", round(accuracy*100, 1), "%)")
print("Precision:", round(precision, 4), "(", round(precision*100, 1), "%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
model.summary()
