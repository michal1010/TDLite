from pathlib import Path

DEFAULT_PROGRAM_DESCRIPTION_FILE = Path("program_description_easy.txt")

PROGRAM_FILE = Path("generated_program.py")
TEST_FILE = Path("test_generated_program.py")
TEST_EXAMPLE = """import unittest
from generated_program import ClassToTest

class Tests(unittest.TestCase):
    # Rename tests, classes and methods based on the program description
    def test_example_1(self):
        class_to_test = ClassToTest()
        self.assertEqual(class_to_test.some_method(), "expected result")

    def test_example_2(self):
        class_to_test = ClassToTest()
        self.assertEqual(class_to_test.some_other_method(), 42)
        
if __name__ == '__main__':
    unittest.main()
"""
MAX_ITERATIONS = 8