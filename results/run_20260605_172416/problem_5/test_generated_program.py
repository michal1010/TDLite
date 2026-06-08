import unittest
from generated_program import intersperse
class Tests(unittest.TestCase):
    def test_empty_list(self):
        result = intersperse([], 4)
        self.assertEqual(result, [])

    def test_with_elements(self):
        result = intersperse([1, 2, 3], 4)
        self.assertEqual(result, [1, 4, 2, 4, 3])
if __name__ == '__main__':
    unittest.main()
