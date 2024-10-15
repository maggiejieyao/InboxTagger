import unittest
from unittest.mock import patch, MagicMock

from models.email_model import *

class TestEmailModel(unittest.TestCase):

    @patch('models.email_model.fetch_user_created_labels')
    def test_fetch_user_created_labels(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().labels().list().execute.return_value = {
            'labels': [
                {'name': 'Label1', 'id': 'label1_id', 'type': 'user'},
                {'name': 'Label2', 'id': 'label2_id', 'type': 'system'}
            ]
        }
        
        labels = fetch_user_created_labels(mock_service)
        
        self.assertEqual(labels, {'Label1': 'label1_id'})
        print("#EM 1. Fetches all user created labels --> PASS")

    @patch('models.email_model.fetch_emails_by_user_created_label')
    def test_fetch_emails_by_user_created_label(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': f'msg{i}'} for i in range(5)]
        }
        mock_service.users().messages().get().execute.side_effect = [
            {'payload': {'body': {'data': 'YWJjZGVm'}, 'headers': [{'name': 'Subject', 'value': 'Test'}]}, 'id': f'msg{i}'} for i in range(5)
        ]
        
        emails = fetch_emails_by_user_created_label(mock_service, 'label_id', max_emails=5)
        
        self.assertEqual(len(emails), 5)
        self.assertEqual(emails[0][1], 'Test')
        print("#EM 2. Fetches all emails with user created labels -->PASS")

    @patch('models.email_model.fetch_emails_by_user_created_label')
    def test_fetch_emails_by_user_created_label_stop_at_max_emails(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': f'msg{i}'} for i in range(10)]
        }
        mock_service.users().messages().get().execute.return_value = {
            'payload': {'body': {'data': 'YWJjZGVm'}, 'headers': [{'name': 'Subject', 'value': 'Test'}]}
        }
        
        emails = fetch_emails_by_user_created_label(mock_service, 'label_id', max_emails=5)
        
        self.assertEqual(len(emails), 5)
        print("#EM3. Fetches user created emails stopped by maximumn parameters --> PASS")

    @patch('models.email_model.fetch_emails_by_user_created_label')
    def test_fetch_emails_by_user_created_label_no_messages(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {'messages': []}
        
        emails = fetch_emails_by_user_created_label(mock_service, 'label_id', max_emails=5)
        
        self.assertEqual(len(emails), 0)
        print("#EM4. Fetches emails stop if no emails with user created email --> PASS")


    @patch('models.email_model.fetch_latest_inbox_emails_with_system_labels')
    def test_fetch_latest_inbox_emails_with_system_labels_stop_at_max_emails(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': f'msg{i}'} for i in range(10)]
        }
        mock_service.users().messages().get().execute.return_value = {
            'payload': {'body': {'data': 'YWJjZGVm'}, 'headers': [{'name': 'Subject', 'value': 'Test'}]}
        }
        
        emails = fetch_latest_inbox_emails_with_system_labels(mock_service, max_emails=5)
        
        self.assertEqual(len(emails), 5)
        print("#EM5. Fetching email with only system labels will stop by maximum parameter --> PASS")

    @patch('models.email_model.fetch_latest_inbox_emails_with_system_labels')
    def test_fetch_latest_inbox_emails_with_system_labels_no_messages(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {'messages': []}
        
        emails = fetch_latest_inbox_emails_with_system_labels(mock_service, max_emails=5)
        
        self.assertEqual(len(emails), 0)
        print("#EM6. Stop fetching if no emails with system labels  --> PASS")    

    @patch('models.email_model.prepare_data')
    def test_prepare_data_no_nan_rows_or_columns(self, mock_build):
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': f'msg{i}'} for i in range(4)]
        }
        mock_service.users().messages().get().execute.return_value = {
            'payload': {'body': {'data': 'YWJjZGVm'}, 'headers': [{'name': 'Subject', 'value': 'Test'}]}
        }
        
        df = prepare_data(mock_service, {'Label1': 'label1_id'})
        
        self.assertFalse(df.isnull().any().any())
        print("#EM7. Emails will saved without any null value --> PASS")

if __name__ == '__main__':
    unittest.main()