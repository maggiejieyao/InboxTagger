import unittest
from unittest.mock import patch, MagicMock

from controllers.email_controller import *

class TestEmailController(unittest.TestCase):

    @patch('controllers.email_controller.apply_label_to_email')
    def test_apply_label_to_email(self, mock_build):
        mock_service = MagicMock()
        email_id = 'test_email_id'
        label_id = 'test_label_id'
        
        apply_label_to_email(mock_service, email_id, label_id)
        
        mock_service.users().messages().modify.assert_called_once_with(
            userId='me',
            id=email_id,
            body={'addLabelIds': [label_id], 'removeLabelIds': []}
        )
        print("EC1.applies labels successfully-->PASS")


if __name__ == '__main__':
    unittest.main()

