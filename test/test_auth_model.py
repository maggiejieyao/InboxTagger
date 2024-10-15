import unittest
from unittest.mock import patch, MagicMock, mock_open

from models.auth_model import authenticate_gmail

class TestAuthModel(unittest.TestCase):
    @patch('models.auth_model.build')
    @patch('models.auth_model.Credentials')
    @patch('models.auth_model.Request')
    @patch('models.auth_model.InstalledAppFlow')
    def test_authenticate_gmail_arguments(self, MockInstalledAppFlow, MockRequest, MockCredentials, MockBuild):
        # Setup mocks
        creds_mock = MagicMock()
        MockCredentials.from_authorized_user_file.return_value = creds_mock
        MockBuild.return_value = 'service_mock'

        service = authenticate_gmail()
        
        self.assertEqual(service, 'service_mock')
        MockBuild.assert_called_once_with('gmail', 'v1', credentials=creds_mock)
        print("#Auth 1. Authentication service with correct credential and build arguments --> PASS")
    
    @patch('models.auth_model.build')
    @patch('models.auth_model.Credentials')
    @patch('models.auth_model.Request')
    @patch('models.auth_model.InstalledAppFlow')
    def test_authenticate_gmail_connection(self, MockInstalledAppFlow, MockRequest, MockCredentials, MockBuild):
        # Setup mock credentials and the service
        creds_mock = MagicMock()
        creds_mock.valid = True  # Indicate that creds are valid
        MockCredentials.from_authorized_user_file.return_value = creds_mock
        MockBuild.return_value = 'service_mock'
        
        # Call the function to test
        service = authenticate_gmail()
        
        # Verify that the service is created and returned correctly
        self.assertEqual(service, 'service_mock')
        MockBuild.assert_called_once_with('gmail', 'v1', credentials=creds_mock)
        print("#Auth 2. Authenication Gmail service connected successfully --> PASS")

    @patch('models.auth_model.build')
    @patch('models.auth_model.Credentials')
    @patch('models.auth_model.Request')
    @patch('models.auth_model.InstalledAppFlow')
    @patch('builtins.open', new_callable=mock_open)
    def test_token_refresh_when_expired(self, mock_open, MockInstalledAppFlow, MockRequest, MockCredentials, MockBuild):
        # Setup mock credentials to simulate an expired token with a refresh token
        creds_mock = MagicMock()
        creds_mock.valid = False
        creds_mock.expired = True
        creds_mock.refresh_token = 'mock_refresh_token'
        creds_mock.to_json.return_value = '{"token": "new_token"}'  # Mock to_json to return a valid JSON string
        
        # Setup mocks for refresh and build
        MockCredentials.from_authorized_user_file.return_value = creds_mock
        MockBuild.return_value = 'service_mock'
        
        # Call the function to test
        service = authenticate_gmail()
        
        # Verify that creds.refresh was called, indicating a token refresh
        creds_mock.refresh.assert_called_once_with(MockRequest())
        
        # Verify that the service was created after the refresh
        self.assertEqual(service, 'service_mock')
        MockBuild.assert_called_once_with('gmail', 'v1', credentials=creds_mock)
        
        # Verify that token.json was written with new credentials
        mock_open.assert_called_once_with('token.json', 'w')
        mock_open().write.assert_called_once_with('{"token": "new_token"}')
        
        print("#Auth 3. Token was refreshed and Gmail service connected --> PASS")
    

if __name__ == '__main__':
    unittest.main()

