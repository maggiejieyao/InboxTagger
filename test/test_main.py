import unittest

from test_auth_model import TestAuthModel
from test_email_model import TestEmailModel
from test_rnn_model import TestRNNModel

from test_email_controller import TestEmailController
from test_view import TestView

# Load test suites
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAuthModel))
    suite.addTest(unittest.makeSuite(TestEmailModel))
    suite.addTest(unittest.makeSuite(TestRNNModel))
    suite.addTest(unittest.makeSuite(TestEmailController))
    suite.addTest(unittest.makeSuite(TestView))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
