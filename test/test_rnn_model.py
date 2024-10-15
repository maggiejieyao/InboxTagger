import unittest
from unittest.mock import patch, MagicMock

from models.rnn_model import *

class TestRNNModel(unittest.TestCase):
    def test_preprocess_text(self):
        text = "Hello World! This is a test, with punctuation and stopwords."
        result = preprocess_text(text)
        
        # Assuming stopwords are removed and text is lemmatized
        self.assertIn("test", result)
        self.assertNotIn("this", result)  # Stopword
        self.assertNotIn("!", result)  # Punctuation
        print(f"After preprocessed, the text with punctuation and stopwords: {result}")
        print("RM1. Preprocess text successfully --> PASS")

    @patch('models.rnn_model.time')
    def test_save_current_timestamp(self, mock_time):
        mock_time.time.return_value = 1234567890.0
        save_current_timestamp()
        
        # Check if timestamp file exists and has correct content
        with open('./sources/last_update_timestamp.txt', 'r') as file:
            content = file.read().strip()
            self.assertEqual(content, '1234567890.0')
        print("RM2. New timestamp saved successfully --> PASS")

    def test_get_last_update_timestamp(self):
        # First test with no file existing
        if os.path.exists('./sources/last_update_timestamp.txt'):
            os.remove('./sources/last_update_timestamp.txt')
        self.assertIsNone(get_last_update_timestamp())
        
        # Now test with the file created
        with open('./sources/last_update_timestamp.txt', 'w') as file:
            file.write(str(1234567890.0))
        timestamp = get_last_update_timestamp()
        self.assertEqual(timestamp, datetime.fromtimestamp(1234567890.0))
        print("RM3. Get last updated timestamp successfully --> PASS")
    
    @patch('models.rnn_model.tf.keras.models.load_model')
    @patch('models.rnn_model.load')
    def test_load_or_train_model_rnn(self, mock_load, mock_load_model):
        # Mock data
        df = pd.DataFrame({
            'body': ['sample text 1', 'sample text 2'],
            'label': ['label1', 'label2']
        })

        # Setup mocks to simulate model and tokenizer loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_load.side_effect = ['tokenizer_mock', 'label_to_code_mock']

        model, tokenizer, label_to_code = load_or_train_model_rnn(df)

        # Check that the mocked objects are returned
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, 'tokenizer_mock')
        self.assertEqual(label_to_code, 'label_to_code_mock')
        print("RM4. Model, tokenizer and label to code loaded --> PASS")
    

if __name__ == '__main__':
    unittest.main()