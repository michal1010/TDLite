import unittest
from generated_program import mean_absolute_deviation

class Tests(unittest.TestCase):
    def test_mean_absolute_deviation_1(self):
        result = mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result, 1.0)

    def test_mean_absolute_deviation_2(self):
        result = mean_absolute_deviation([5.0, 7.0, 9.0, 11.0])
        self.assertEqual(result, 2.0)

if __name__ == '__main__':
    unittest.main()
