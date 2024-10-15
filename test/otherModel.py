from googleapiclient.discovery import build
import base64
from google.oauth2.credentials import Credentials
import pandas as pd
import os
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from joblib import dump,load
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import csv
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Google API Scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Download NLTK resources if not already available
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
#app = Flask(__name__)

def authenticate_gmail():
    """Authenticate and connect to the Gmail API."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service


# Fetch only user-created labels
def fetch_user_created_labels(service):
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    
    # Filter out system labels
    user_created_labels = [label for label in labels if label['type'] == 'user']
    
    # Create a mapping of label names to their respective IDs
    label_mapping = {label['name']: label['id'] for label in user_created_labels}
   
    return label_mapping

# Fetch emails from a specific label
def fetch_emails_by_label(service, label_id, label_name, max_emails=500):
    emails = []
    page_token = None  # Start with no page token
    total_fetched = 0  # To track the total number of emails fetched for this label

    while total_fetched < max_emails:
        # Make a request to list messages for the given label, using the page token if available
        results = service.users().messages().list(userId='me', labelIds=[label_id], maxResults=100, pageToken=page_token).execute()
        messages = results.get('messages', [])
        
        # If no messages are returned, break the loop
        if not messages:
            break
        
        print(f"Fetched {len(messages)} messages for label {label_name} in this batch.")
        
        for msg in messages:
            if total_fetched >= max_emails:
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
            emails.append((msg['id'], subject, email_body, label_name))
            total_fetched += 1  # Increment the total number of emails fetched for this label
        
        # Check if there's a next page, if not, break the loop
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break  # No more pages to fetch for this label

    print(f"Fetched a total of {len(emails)} emails for label {label_name}.")
    
    return emails

# Function to calculate and plot the similarity between each label
def calculate_label_similarity(df):
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    # Compute TF-IDF vectors for each email body
    tfidf_matrix = tfidf.fit_transform(df['body'])
    
    # Group the TF-IDF vectors by label and calculate the average vector for each label
    label_groups = df.groupby('label')['body'].apply(lambda texts: ' '.join(texts))
    label_tfidf_matrix = tfidf.transform(label_groups)
    
    # Calculate cosine similarity between labels
    similarity_matrix = cosine_similarity(label_tfidf_matrix)
    
    # Convert to DataFrame for easier plotting
    labels = label_groups.index.tolist()
    similarity_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    
    # Plot the similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Label Similarity Matrix")
    plt.xlabel("Labels")
    plt.ylabel("Labels")
    plt.show()

# Preprocess the email text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # Tokenize and remove stopwords, and lemmatize each token
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Rejoin into a single string
    return ' '.join(words)

# Prepare the data by creating a DataFrame for emails under user-created labels and export to CSV
def prepare_data(service, user_label_mapping):
    data = []
    
    for label_name, label_id in user_label_mapping.items():
        emails = fetch_emails_by_label(service, label_id, label_name)
        for email_id, subject, email_body, label_name in emails:
            data.append({
                'email_id': email_id,
                'subject': subject,
                'body': email_body,
                'label': label_name
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df


# Check if model and vectorizer exist, otherwise train the Random Forest model with hyperparameter tuning
def load_or_train_model_rf(df):
    label_to_code = load('label_to_code.pkl') if os.path.exists('label_to_code.pkl') else None
    
    # Convert labels to categorical format
    df['label'] = pd.Categorical(df['label'])
    label_to_code = dict(enumerate(df['label'].cat.categories))  # Maps 0 -> 'Work', 1 -> 'Personal', etc.
    
    df['label'] = df['label'].cat.codes
    y = np.array(df['label'].values)

    X_train, X_test, y_train, y_test = train_test_split(df['body'].values, y, test_size=0.2, random_state=42)

    # Initialize TF-IDF vectorizer with n-grams
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)


    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        # Fit the grid search to the training data
        model.fit(X_train_tfidf, y_train)
        # Evaluate the model on the test set
        y_pred = model.predict(X_test_tfidf)
        end_time = time.time()
        accuracy = accuracy_score(y_test, y_pred)
        processing_time = end_time - start_time
        class_report = classification_report(y_test, y_pred)

        results[model_name] = {
            'Accuracy': accuracy,
            'Classification Report': class_report,
            'Processing Time': processing_time
        }

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:\n", class_report)
        print(f"Processing Time: {processing_time:.2f} seconds\n")

    return results

# Fetch primary inbox emails with subject and handle None for body
def fetch_primary_inbox_emails(service, max_emails=300):
    emails = []
    page_token = None  # Start with no page token
    total_fetched = 0  # To track the total number of emails fetched
    
    while total_fetched < max_emails:
        # Make a request to list messages with INBOX label, using the page token if available
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=100, pageToken=page_token).execute()
        messages = results.get('messages', [])
        
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

            # Append email ID, subject, and body (emails with only INBOX label)
            emails.append((msg['id'], subject, email_body))
            total_fetched += 1  # Increment the total number of emails fetched
        
        # Check if there's a next page, if not, break the loop
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

    print(f"Fetched a total of {len(emails)} primary inbox emails with only the INBOX label.")
    
    return emails

if __name__ == '__main__':
    
    service = authenticate_gmail()

    # Fetch user-created labels
    user_label_mapping = fetch_user_created_labels(service)

    # Prepare the data and either load or train the SVM model
    df = prepare_data(service, user_label_mapping)
    #calculate_label_similarity(df)

    load_or_train_model_rf(df)

