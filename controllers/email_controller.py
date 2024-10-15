from flask import render_template, request, jsonify
from models.auth_model import authenticate_gmail
from models.email_model import prepare_data, fetch_user_created_labels
from models.rnn_model import load_or_train_model_rnn, update_model_if_needed
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SYSTEM_LABEL_EMAILS = 10

# Apply a Gmail label to a specific email
def apply_label_to_email(service, email_id, label_id):
    try:
        service.users().messages().modify(
            userId='me',
            id=email_id,
            body={
                'addLabelIds': [label_id],
                'removeLabelIds': []
            }
        ).execute()
        print(f"Label with ID {label_id} successfully applied to email {email_id}")
    except Exception as e:
        print(f"An error occurred while applying the label to email {email_id}: {str(e)}")

# Fetch emails, predict labels, and apply the appropriate labels
def predict_and_label_emails(service, model, tokenizer, user_label_mapping, label_to_code, df, max_emails):
    if label_to_code is None:
        print("Error: label_to_code is None. Cannot map predicted labels to Gmail labels.")
        return

    if df.empty:
        print("No emails found for processing.")
        return

    # Tokenize and pad the email bodies for prediction
    X_seq = tokenizer.texts_to_sequences(df['body'])
    X_padded = pad_sequences(X_seq, maxlen=200)

    # Predict the labels for the emails
    predictions = model.predict(X_padded)
    predicted_labels = np.argmax(predictions, axis=1)

    total_fetched = 0

    while total_fetched < max_emails:
        if total_fetched >= max_emails:
            break
        for email_id, pred, subject in zip(df['email_id'], predicted_labels, df['subject']):
            label_name = label_to_code.get(pred, 'system_label_only')
            print(f"Email Subject: '{subject}' | Predicted Label: '{label_name}'")
            if label_name in user_label_mapping:
                label_id = user_label_mapping[label_name]
                apply_label_to_email(service, email_id, label_id)
                total_fetched+=1
                print(f"Labeled Email ID {email_id} with '{label_name}'")
                if total_fetched >= max_emails:
                    break
            else:
                print(f"No Gmail label found for the predicted label '{label_name}'")

def predict_emails():
    service = authenticate_gmail()

    # Fetch user-created labels
    user_label_mapping = fetch_user_created_labels(service)

    # Prepare the data and either load or train the RNN model
    df = prepare_data(service, user_label_mapping)
    rnn_model, tokenizer, label_to_code = load_or_train_model_rnn(df)

    #Check if model needs updating and update if necessary
    update_model_if_needed(service, user_label_mapping, rnn_model, tokenizer, label_to_code)
    
    # Predict and label the emails
    predict_and_label_emails(service, rnn_model, tokenizer, user_label_mapping, label_to_code, df, max_emails=MAX_SYSTEM_LABEL_EMAILS)

    return jsonify({"status": "Emails processed and labeled successfully!"})

def home():
    return render_template('index.html')