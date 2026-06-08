import unittest
from generated_program import below_zero

class Tests(unittest.TestCase):
    def test_balance_never_falls_below_zero(self):
        self.assertFalse(below_zero([1, 2, 3]))

    def test_balance_falls_below_zero(self):
        self.assertTrue(below_zero([1, 2, -4, 5]))

if __name__ == '__main__':
    unittest.main()
