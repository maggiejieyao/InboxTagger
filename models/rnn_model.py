import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
import os
import tensorflow as tf
import pandas as pd
import time
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import base64
import keras_tuner as kt

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# File to save the timestamp of the last model update
TIMESTAMP_FILE = './sources/last_update_timestamp.txt'
MODEL_PATH = './sources/rnn_model.h5'
TOKENIZER_PATH = './sources/tokenizer.pkl'
LABELS_TO_CODE_PATH = './sources/label_to_code.pkl'
MAX_USER_LABEL_EMAILS =  200
MAX_FETCHED = 10

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Define the model-building function for Keras Tuner
def build_model(hp, num_classes):
    model = Sequential()

    # Hyperparameter choices for the Embedding layer
    embedding_dim = hp.Choice('embedding_dim', values=[32, 64, 128], default=64)
    model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=200))

    # First LSTM layer with return_sequences=True
    model.add(LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1)))

    # Second LSTM layer without return_sequences
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)))

    # Fully connected layer
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dense_dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Output layer with num_classes units
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load or train the RNN model
def load_or_train_model_rnn(df):

    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading existing RNN model and tokenizer...")
        model = tf.keras.models.load_model(MODEL_PATH)
        tokenizer = load(TOKENIZER_PATH)
        label_to_code = load(LABELS_TO_CODE_PATH) if os.path.exists(LABELS_TO_CODE_PATH) else None
    else:
        print("No existing model found. Training new RNN model...")

        # Convert labels to categorical format
        df['label'] = pd.Categorical(df['label'])
        label_to_code = dict(enumerate(df['label'].cat.categories))
        
        df['label'] = df['label'].cat.codes
        y = np.array(df['label'].values)
        y = to_categorical(y)

        # Split data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(df['body'].values, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        # Initialize Tokenizer and convert text to sequences
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>", char_level=False)
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_val_seq = tokenizer.texts_to_sequences(X_val)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        # Pad sequences to ensure uniform input size
        X_train_padded = pad_sequences(X_train_seq, maxlen=200)
        X_val_padded = pad_sequences(X_val_seq, maxlen=200)
        X_test_padded = pad_sequences(X_test_seq, maxlen=200)

        model = Sequential([
            Embedding(input_dim=10000, output_dim=64, input_length=200),
            LSTM(64, return_sequences=True),
            Dense(64, activation='relu')
        ])

        '''
        # Build the RNN model
        model = Sequential([
            Embedding(input_dim=10000, output_dim=64, input_length=200),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(y.shape[1], activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''
        num_classes = y.shape[1]
        # Initialize Keras Tuner and search for the best hyperparameters
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, num_classes),
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory='./sources/hyperparameter_tuning',
            project_name='InboxTagger'
        )

        # Run the search
        tuner.search(X_train_padded, y_train, epochs=10, validation_data=(X_val_padded, y_val), batch_size=32)

        # Get the best hyperparameters and model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)

        # Train the model with the best hyperparameters
        start_time = time.time()
        model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_val_padded, y_val))
        end_time = time.time()

        '''
        start_time = time.time()
        # Train the model with validation data
        model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
        end_time = time.time()
        '''

        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(X_test_padded, y_test)
        processing_time = end_time - start_time
        print("RNN Model:")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Classification Report:\n{classification_report(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test_padded), axis=1), zero_division=0)}")
        print(f"Processing Time: {processing_time:.2f} seconds\n")

        # Save the trained model and tokenizer
        model.save(MODEL_PATH)
        dump(tokenizer, TOKENIZER_PATH)
        dump(label_to_code, LABELS_TO_CODE_PATH)
        save_current_timestamp()
    return model, tokenizer, label_to_code

# Update timestamp after re-access the application
def get_last_update_timestamp():
    """Retrieve the last model update timestamp from file or return None if it doesn't exist."""
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, 'r') as file:
            timestamp = float(file.read().strip())
            return datetime.fromtimestamp(timestamp)
    return None

# Save current timestamp after accessing
def save_current_timestamp():
    current_timestamp = time.time()
    with open(TIMESTAMP_FILE, 'w') as file:
        file.write(str(current_timestamp))

# Fetch emails after the last update timestamp
def fetch_emails_after_timestamp(service, label_id, last_update=None, max_emails=MAX_USER_LABEL_EMAILS):
    emails = []
    page_token = None
    total_fetched = 0

    while total_fetched < max_emails:
        results = service.users().messages().list(userId='me', labelIds=[label_id], maxResults=MAX_FETCHED, pageToken=page_token).execute()
        messages = results.get('messages', [])
        
        if not messages:
            break
        
        for msg in messages:
            if total_fetched >= max_emails:
                break

            message = service.users().messages().get(userId='me', id=msg['id']).execute()
            
            email_body = ""
            subject = "(No Subject)"
            timestamp = int(message['internalDate']) / 1000.0
            email_date = datetime.fromtimestamp(timestamp)

            # Fetch only emails received after the last update
            if last_update and email_date <= last_update:
                continue

            headers = message.get('payload', {}).get('headers', [])
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                    break

            if 'data' in message['payload']['body']:
                email_body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

            # Determine language and apply corresponding preprocessing
            email_body = preprocess_text(email_body) 

            emails.append((msg['id'], subject, email_body, label_id))
            total_fetched += 1

        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

    return emails

# Update the model with new data if more than a month has passed since the last update
def update_model_if_needed(service, user_label_mapping, rnn_model, tokenizer, label_to_code):
    last_update = get_last_update_timestamp()
    one_month_ago = datetime.now() - timedelta(days=30)

    # If last update was more than a month ago, fetch new emails and retrain the model
    if not last_update or last_update < one_month_ago:
        print("Updating model with new data...")

        new_data = []
        for label_name, label_id in user_label_mapping.items():
            new_emails = fetch_emails_after_timestamp(service, label_id, last_update)
            for email_id, subject, email_body, label_id in new_emails:
                new_data.append({
                    'email_id': email_id,
                    'subject': subject,
                    'body': preprocess_text(email_body),
                    'label': label_name
                })

        # If there are new emails, convert to DataFrame and update the model
        if new_data:
            new_df = pd.DataFrame(new_data)
            rnn_model, tokenizer, label_to_code = load_or_train_model_rnn(new_df)
            save_current_timestamp()  # Save current timestamp as the last update time
            print("Model updated with new data.")
        else:
            print("No new emails found since the last update.")
    else:
        print("Model update not needed. Last update was within a month.")


