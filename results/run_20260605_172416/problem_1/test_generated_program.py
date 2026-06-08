import unittest
from generated_program import separate_paren_groups

class Tests(unittest.TestCase):
    def test_example_1(self):
        self.assertEqual(separate_paren_groups('( ) (( )) (( )( ))'), ['()', '(())', '(()())'])

if __name__ == '__main__':
    unittest.main()
