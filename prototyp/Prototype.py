import os
import signal
import sqlite3
import webbrowser
from threading import Timer
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras

import dash
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output, State
import dash.exceptions

# Get the directory:
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSV file:
csv_path = os.path.join(current_dir, 'datasets', 'data.csv')

def load_and_preprocess_data(csv_path):
    try:
        data = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: The file 'data.csv' was not found at {csv_path}")
        print("Current working directory:", os.getcwd())
        print("Please make sure the file exists and is in the correct location.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)

    data = data.drop(columns=['Date', 'Value_date', 'IBAN', 'Balance'])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop(columns=['Category'])
    y = data['Category']

    # Create a text field from Reference and Recipient/Sender
    X['text'] = X['Reference'].astype(str) + " " + X['Recipient/Sender'].astype(str)
    X['Amount'] = X['Amount'].astype(float)
    X_encoded = pd.get_dummies(X['Transaction_Type'])
    X_numeric = X[['Amount']]

    vectorizer = CountVectorizer(max_features=1000, analyzer='char')
    X_text = vectorizer.fit_transform(X['text'])

    X_numeric['Amount_sign'] = np.sign(X_numeric['Amount'])
    
    X_combined = hstack([X_text, X_numeric, X_encoded])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5% of the data is used for adding later in the P&L database
    X_combined_train, X_test, y_onehot, y_test = train_test_split(X_combined, y_onehot, test_size=0.05, random_state=42)

    shapes = {
        'X_text': X_text.shape,
        'X_numeric': X_numeric.shape,
        'X_encoded': X_encoded.shape,
    }
 
    return X_combined_train, y_onehot, shapes, label_encoder, data, X_test, y_test, X_combined

def create_and_train_models(X_combined, y_onehot, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_onehot, test_size=0.1, random_state=42)

    model_path = os.path.join(current_dir, 'model.pkl')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = keras.Sequential([
            keras.layers.Reshape((X_combined.shape[1], 1), input_shape=(X_combined.shape[1],)),
            keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(y_onehot.shape[1], activation='softmax')
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        kfold = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kfold.split(X_train):
            X_train_kfold, X_val_kfold = X_train[train_index], X_train[test_index]
            y_train_kfold, y_val_kfold = y_train[train_index], y_train[test_index]
            model.fit(X_train_kfold, y_train_kfold, epochs=50, batch_size=32, verbose=1, validation_data=(X_val_kfold, y_val_kfold))

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    pred_prob = model.predict(X_test)
    X_test_dense = X_test.toarray()
    # Maybe use for logic decision
    amount = X_test_dense[:, -5]
    revenue_idx = np.where(label_encoder.classes_ == "Revenue")[0][0]

    pred = np.argmax(pred_prob, axis=1)
    accuracy = np.mean(pred == np.argmax(y_test, axis=1))

    return accuracy, model

def make_PandL_database(model, X_combined, shapes, label_encoder):
    X_text_shape = shapes['X_text'][1]
    X_numeric_shape = shapes['X_numeric'][1]

    pred = np.argmax(model.predict(X_combined), axis=1)

    X_combined_dense = X_combined.toarray()
    X_numeric = X_combined_dense[:, X_text_shape:X_text_shape+X_numeric_shape]

    amount = X_numeric[:, 0]
    labels = label_encoder.inverse_transform(pred)

    create_database(label_encoder.classes_)
    for label, amount in zip(labels, amount):
        add_prediction(label, amount)

def add_prediction(label, new_amount):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (label, amount) 
        VALUES (?, ?)
        ON CONFLICT(label) DO UPDATE SET amount = amount + excluded.amount
    """, (label, new_amount))
    conn.commit()
    conn.close()

def create_database(labels):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("""
        CREATE TABLE predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE,
            amount REAL DEFAULT 0.0
        )
    """)
    for label in labels:
        cursor.execute("INSERT OR IGNORE INTO predictions (label) VALUES (?)", (label,))
    conn.commit()
    conn.close()

# Load and preprocess data
X_combined_train, y_onehot, shapes, label_encoder, df, X_test, y_test, X_combined = load_and_preprocess_data(csv_path)

# Create and train models
accuracy, model = create_and_train_models(X_combined_train, y_onehot, label_encoder)

# Make P&L database
make_PandL_database(model, X_combined, shapes, label_encoder)


# Create P&L-Table of the dataset without predictions
pl_data = df.groupby('Category')['Amount'].sum().reset_index()
pl_data = pl_data.sort_values('Amount', ascending=False)

pl_figure = dash_table.DataTable(
    columns=[
        {'name': 'Category', 'id': 'Category'},
        {'name': 'Amount', 'id': 'Amount', 'type': 'numeric', 'format': {'specifier': ',.2f'}}
    ],
    data=pl_data.assign(Amount=pl_data['Amount'].round(2)).to_dict('records'),
    style_table={'overflowX': 'auto'}
)
def load_initial_predictions():
    """ Fetch predictions from the database at startup """
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT label, amount FROM predictions ORDER BY amount DESC")
    rows = cursor.fetchall()
    conn.close()

    return [{'category': row[0], 'amount': round(row[1], 2)} for row in rows]
# Initialize Dash-App
app = dash.Dash(__name__, assets_folder='assets')

# Define Layout
app.layout = html.Div([
    html.Div([
        html.H2('Data Preview (First 5 Rows):'),
        dash_table.DataTable(
            data=df.head().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'}
        )
    ]),
    html.Div([
        html.H2('P&L Statement with actual categories:'),
        html.Div(id='pl-table', children=[pl_figure])
    ]),
    # html.Div([
    #     html.H2('Random Classifier:'),
    #     html.P(f'Theoretical Accuracy: {1/7:.3f}'),
    #     html.H2('Model Accuracies:'),
    #     html.P(f'Model Accuracy: {accuracy:.3f}')
    html.Div([
        html.H2('P&L Statement with predicted categories:', style={'display': 'inline-block', 'verticalAlign': 'middle'}),
        html.Button('Add new Transaction', id='add-transaction-button', n_clicks=0,
                    style={'fontSize': '20px', 'padding': '10px 20px', 'display': 'inline-block', 'margin-left': '20px', 'verticalAlign': 'middle'}),
        html.Button('Quit', id='quit-button', n_clicks=0,
                    style={'fontSize': '20px', 'padding': '10px 20px', 'display': 'inline-block', 'margin-left': '10px', 'verticalAlign': 'middle'})
    ], style={'textAlign': 'left', 'margin-bottom': '20px'}),
    html.Div(id='predictions-table', children=[dash_table.DataTable(
        columns=[{'name': 'Category', 'id': 'category'},
                 {'name': 'Amount', 'id': 'amount', 'type': 'numeric', 'format': {'specifier': ',.2f'}}],
        data=load_initial_predictions(),
        style_table={'overflowX': 'auto'}
    )]),
    html.Div(id='hidden-div', style={'display': 'none'}),
    # The confirm dialog that will display the full transaction details
    dcc.ConfirmDialog(id='transaction-confirm-dialog'),
    # Store to hold the details of the transaction until confirmation
    dcc.Store(id='transaction-data')
])
# -------------------------------
# Callback to prepare the transaction and show details in an alert dialog
# -------------------------------
@app.callback(
    [Output('transaction-data', 'data'),
     Output('transaction-confirm-dialog', 'displayed'),
     Output('transaction-confirm-dialog', 'message')],
    [Input('add-transaction-button', 'n_clicks')]
)
def prepare_transaction(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    # Convert the test set to a dense array
    X_test_dense = X_test.toarray()
    # Get model predictions for all test transactions
    pred_all = np.argmax(model.predict(X_test), axis=1)
    # Randomly select a transaction index
    idx = np.random.randint(len(X_test_dense))
    new_transaction = X_test_dense[idx]
    new_transaction_text_shape = shapes['X_text'][1]
    new_transaction_numeric_shape = shapes['X_numeric'][1]
    new_transaction_numeric = new_transaction[new_transaction_text_shape:new_transaction_text_shape+new_transaction_numeric_shape]
    amount = new_transaction_numeric[0]
    predicted_label = label_encoder.inverse_transform([pred_all[idx]])[0]
    # Get real category from y_test (which is one-hot encoded)
    real_idx = np.argmax(y_test[idx])
    real_label = label_encoder.inverse_transform([real_idx])[0]
    # Build the alert message with transaction details
    message = (f"Transaction Details:\n"
               f"Amount: {amount:.2f}\n"
               f"Real Category: {real_label}\n"
               f"Predicted Category: {predicted_label}\n\n"
               "Click OK to add this transaction.")
    # Store the transaction details so they can be used upon confirmation
    transaction_details = {
        'amount': float(amount),
        'predicted_label': predicted_label,
        'real_label': real_label
    }
    return transaction_details, True, message

# -------------------------------
# Callback to add the transaction after the user confirms in the dialog
# -------------------------------
@app.callback(
    Output('predictions-table', 'children'),
    [Input('transaction-confirm-dialog', 'submit_n_clicks')],
    [State('transaction-data', 'data')]
)
def add_transaction(confirm_clicks, transaction_details):
    if not confirm_clicks or transaction_details is None:
        raise dash.exceptions.PreventUpdate
    # Add the transaction using the predicted label and amount
    add_prediction(transaction_details['predicted_label'], transaction_details['amount'])
    # After adding, read the updated predictions from the database and rebuild the table
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY amount DESC")
    rows = cursor.fetchall()
    conn.close()

    table = dash_table.DataTable(
        columns=[
            {'name': 'Category', 'id': 'category'},
            {'name': 'Amount', 'id': 'amount', 'type': 'numeric', 'format': {'specifier': ',.2f'}}
        ],
        data=[{'category': row[1], 'amount': round(row[2], 2)} for row in rows],
        style_table={'overflowX': 'auto'}
    )

    return table

# -------------------------------
# Callback to handle the Quit button
# -------------------------------
@app.callback(
    Output('hidden-div', 'children'),
    [Input('quit-button', 'n_clicks')]
)
def handle_quit(quit_clicks):
    if quit_clicks and quit_clicks > 0:
        os.kill(os.getpid(), signal.SIGINT)
    return ""

# Open the browser automatically
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=False)
