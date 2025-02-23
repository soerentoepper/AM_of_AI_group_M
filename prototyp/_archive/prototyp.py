import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

# Load data from datasets/bank1.csv
data = pd.read_csv('prototyp/datasets/bank1.csv', delimiter=';')

# Drop columns
data = data.drop(columns=['Account No', 'DATE', 'VALUE DATE'])

# Convert columns to numeric
data['WITHDRAWAL AMT'] = pd.to_numeric(data['WITHDRAWAL AMT'].str.replace(',', '').str.replace('.', ''))
data['DEPOSIT AMT'] = pd.to_numeric(data['DEPOSIT AMT'].str.replace(',', '').str.replace('.', ''))
data['BALANCE AMT'] = pd.to_numeric(data['BALANCE AMT'].str.replace(',', '').str.replace('.', ''))
data['CHQ.NO.'] = pd.to_numeric(data['CHQ.NO.'], errors='coerce')

# Convert Transaction detail to string
data['TRANSACTION DETAILS'] = data['TRANSACTION DETAILS'].astype(str)

# Split data in features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
text_features = [0]
train_data = Pool(data=X_train, label=y_train, text_features=text_features)

test_data = Pool(data=X_test, label=y_test, text_features=text_features)

# Train model
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=10
)

model.fit(train_data)

# Evaluate model
predictions = model.predict(X_test)

# Compare the predictions with the actual values
score_list = []
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        score_list.append(1)
    else:
        score_list.append(0)

print(f"Number of samples in test set: {len(score_list)}")
print(f"Number of correct predictions: {sum(score_list)}")
print(f"Accuracy: {sum(score_list) / len(score_list)}")

