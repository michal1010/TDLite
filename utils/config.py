import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROGRAM_DESCRIPTION_FILE = Path("program_description_easy.txt")
PROGRAM_FILE = Path("outputs/generated_program.py")
TEST_FILE    = Path("outputs/generated_tests.py")

MODEL_NAME    = os.getenv("MODEL", "qwen2.5-coder:7b")
NUM_CTX       = 8192
TEMP_GENERATE = 0.2   # A1, A2 
TEMP_DEBUG    = 0.2   # A4    

MAX_ITERATIONS = 5
MAX_COMPILE_ITERATIONS = 5
MAX_TEST_ITERATIONS = 5

TEST_EXAMPLE = """\
import unittest
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