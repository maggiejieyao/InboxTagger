import unittest
from app import app

class TestView(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route_renders_index_html(self):
        # Send a GET request to the home route
        response = self.app.get('/')
        
        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        
        # Check if 'index.html' content is in the response
        self.assertIn(b'<title>', response.data)  
        self.assertIn(b'Click the button below to run the prediction on your Gmail inbox', response.data)  
        
        print("TR1. Index page rendered successfully --> PASS")

if __name__ == '__main__':
    unittest.main()