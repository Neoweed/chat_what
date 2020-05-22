import unittest
from serv import app
class TestBot(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()
    def test_bot(self):
        rv = self.app.get('/bot')
        self.assertEqual(rv.status, '200 OK')
        self.assertEqual(rv.data, b'Hello World!\n')
if __name__ == '__main__':
    import xmlrunner
    runner = xmlrunner.XMLTestRunner(output='test-reports')
    unittest.main(testRunner=runner)