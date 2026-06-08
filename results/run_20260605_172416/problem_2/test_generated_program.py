import unittest
from generated_program import truncate_number

class Tests(unittest.TestCase):
    def test_truncate_number_1(self):
        self.assertEqual(truncate_number(3.5), 0.5)

    def test_truncate_number_2(self):
        self.assertEqual(truncate_number(10.99), 0.99)

    def test_truncate_number_3(self):
        self.assertEqual(truncate_number(0.75), 0.75)

if __name__ == '__main__':
    unittest.main()
