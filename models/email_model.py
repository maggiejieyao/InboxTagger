import os
import pandas as pd
from datetime import datetime, timedelta
from models.rnn_model import MODEL_PATH, TOKENIZER_PATH, LABELS_TO_CODE_PATH, preprocess_text
import base64


SYSTEM_LABELS = ['INBOX', 'READ', 'UNREAD', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS', 'CATEGORY_PRIMARY']
MAX_USER_LABEL_EMAILS = 200
MAX_FETCHED = 100
MAX_SYSTEM_LABEL_EMAILS = 10

# Fetch only user-created labels
def fetch_user_created_labels(service):
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    
    # Filter out system labels
    user_created_labels = [label for label in labels if label['type'] == 'user']
    #print(user_created_labels)
    # Create a mapping of label names to their respective IDs
    label_mapping = {label['name']: label['id'] for label in user_created_labels}
    #print(user_created_labels)
    return label_mapping

# Fetch emails for a specific user-created label, limited to max_emails
def fetch_emails_by_user_created_label(service, label_id, max_emails=MAX_USER_LABEL_EMAILS):
    emails = []
    page_token = None  # Start with no page token
    total_fetched = 0  # To track the total number of emails fetched

    while total_fetched < max_emails:
        # Make a request to list messages for the given label, using the page token if available
        results = service.users().messages().list(userId='me', labelIds=[label_id], maxResults=MAX_FETCHED, pageToken=page_token).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print("Error: No email with user created label.")
            break  # Exit if no more messages are returned
        
        print(f"Fetched {len(messages)} messages for user-created label in this batch.")
        
        
        for msg in messages:
            if total_fetched >= max_emails:
                print(f"total fecthed inner:{total_fetched}")
                
                break  # Stop fetching if we already reached the max limit for this label
            
            message = service.users().messages().get(userId='me', id=msg['id']).execute()

            email_body = ""
            subject = "(No Subject)"
            
            # Fetch email subject from headers
            headers = message.get('payload', {}).get('headers', [])
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                    break

            # Fetch email body
            if 'data' in message['payload']['body']:
                email_body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

            # Append email ID, subject, body, and label name
            emails.append((msg['id'], subject, email_body, label_id))
            total_fetched += 1  # Increment the total number of emails fetched for this label
        
        # Check if there's a next page, if not, break the loop
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break  # No more pages to fetch for this label
        

    print(f"Fetched a total of {len(emails)} emails for user-created label.")
    
    return emails


# Prepare the data by fetching emails under user-created labels and the latest system-labeled emails.
def prepare_data(service, user_label_mapping=None):
    data = []
    
    # Fetch the latest 10 system-labeled emails
    print("Fetching the latest 10 system-labeled emails (excluding SPAM)...")
    system_label_emails = fetch_latest_inbox_emails_with_system_labels(service, max_emails=MAX_SYSTEM_LABEL_EMAILS)
    
    # Add system-labeled emails to the data
    for email_id, subject, email_body, label_ids in system_label_emails:
        data.append({
            'email_id': email_id,
            'subject': subject,
            'body': preprocess_text(email_body),
            'label': 'system_label'  # Temporarily mark as system label
        })

    # Check if the model exists
    model_exists = os.path.exists(MODEL_PATH)
    label_to_code_exists = os.path.exists(LABELS_TO_CODE_PATH)

    if not model_exists or not label_to_code_exists:
        # Fetch emails under user-created labels only if the model needs to be trained
        print("No trained model found. Fetching emails under user-created labels...")
        if not user_label_mapping:
            user_label_mapping = fetch_user_created_labels(service)
        
        # Fetch and process emails for each user-created label
        for label_name, label_id in user_label_mapping.items():
            print(f"Fetching emails for label: {label_name}")
            user_created_emails = fetch_emails_by_user_created_label(service, label_id, max_emails=MAX_USER_LABEL_EMAILS)
            for email_id, subject, email_body, label_id in user_created_emails:
                data.append({
                    'email_id': email_id,
                    'subject': subject,
                    'body': preprocess_text(email_body),
                    'label': label_name
                })
                
    else:
        print("Model exists. Only fetching system-labeled emails for prediction.")
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    return df

# Check if the email has only system labels (e.g., INBOX, READ, excluding SPAM)
def is_only_system_labels(label_ids):
    for label_id in label_ids:
        if label_id not in SYSTEM_LABELS:
            return False
    return True

# Fetch the latest emails with only system labels (excluding SPAM), limited to max_emails
def fetch_latest_inbox_emails_with_system_labels(service, max_emails=MAX_SYSTEM_LABEL_EMAILS):
    emails = []
    page_token = None  # Start with no page token
    total_fetched = 0  # To track the total number of emails fetched
    
    while total_fetched < 10:
        # Make a request to list messages with INBOX label, using the page token if available
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=MAX_FETCHED, pageToken=page_token).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print("Error: no email with system label.")
            break  # Exit if no more messages are returned
        
        print(f"Fetched {len(messages)} messages in this batch.")
        
        for msg in messages:
            message = service.users().messages().get(userId='me', id=msg['id']).execute()
            
            email_body = ""
            subject = "(No Subject)"
            
            # Fetch email subject from headers
            headers = message.get('payload', {}).get('headers', [])
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                    break

            # Fetch email body
            if 'data' in message['payload']['body']:
                email_body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

            # Ensure labelIds exists, else set to an empty list
            label_ids = message.get('labelIds', [])

            # Only process emails that have system labels (excluding SPAM)
            if is_only_system_labels(label_ids):
                # Append email ID, subject, body, and labels
                emails.append((msg['id'], subject, email_body, label_ids))
                total_fetched += 1  # Increment the total number of emails fetched
                #print(f"total fecched: {total_fetched}")
                #print(f"max email{max_emails}")
                if total_fetched >= 10:
                    break  # Stop fetching if we have reached the max limit

        # Check if there's a next page, if not, break the loop
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

    print(f"Fetched a total of {len(emails)} system label-only emails.")
    
    return emails

