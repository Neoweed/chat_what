# project/test_basic.py
 
 
import os
import unittest
 
from serv import app
 
 

 
 
class BasicTests(unittest.TestCase):
 
###############
#### tests ####
###############
 
    def test_main_page(self):
        tester = app.test_client(self)
        response = tester.get('/bot')
        self.assertEqual(response.status_code, 200)
 
 
if __name__ == "__main__":
    unittest.main()
