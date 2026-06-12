# =============================================================================
# AGENT 1 — Requirement Analyst
# Input:  raw HumanEval or MBPP problem statement
# Output: structured natural language test descriptions as {description, input, expected_output} triplets
# =============================================================================

TEST_1_SYSTEM_PROMPT = """
You are a test analyst. You will receive a programming problem in either HumanEval format
(Python function signature + docstring + >>> examples), 
along with an example of conversion of a problem statement into structured test descriptions.
Your goal is to produce one test case per distinct behavior, in the following format:

Test case N:
  Description: <one sentence describing what this test verifies>
  Input: <concrete input values>
  Expected output: <exact expected return value>

Rules:
1. HumanEval: derive cases strictly from the >>> examples and docstring constraints. Do not invent new cases.
2. MBPP: derive the minimal set of concrete, representative cases from the problem description.
3. Each field must be on its own line. Never compress a test case onto one line.
4. Inputs must include variable names (e.g. cost = [[1, 2, 3]], m = 2, n = 2 — not [[1, 2, 3]], 2, 2).
5. Inputs and outputs must be concrete literals, not descriptions (e.g. [1.0, 2.0] not "a list of floats").
6. Group related cases with a short comment (e.g. # Cases: returns False).
"""

TEST_1_EXAMPLE_1 = """
# EXAMPLE INPUT 1

from typing import List
def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"\

# EXAMPLE OUTPUT 1
# # Cases: typical grouped input
Test case 1:
  Description: Splits a string of three balanced paren groups, ignoring spaces.
  Input: paren_string = '( ) (( )) (( )( ))'
  Expected output: ['()', '(())', '(()())']
"""


TEST_1_EXAMPLE_2 = """
# EXAMPLE INPUT 2
Write a function to find the minimum cost path to reach (m, n) from (0, 0) in a grid.


# EXAMPLE OUTPUT 2
# Cases: small grid
Test case 1:
  Description: Finds minimum cost path in a 3x3 grid with uniform cost.
  Input: grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]], m = 2, n = 2
  Expected output: 5

# Cases: single cell
Test case 2:
  Description: Returns the cell value when start and end are the same cell.
  Input: grid = [[7]], m = 0, n = 0
  Expected output: 7

# Cases: single row
Test case 3:
  Description: Traverses a single row by summing all elements.
  Input: grid = [[1, 2, 3]], m = 0, n = 2
  Expected output: 6
"""



# =============================================================================
# AGENT 2 — Test Code Generator
# Input:  (1) original problem statement [as context]
#         (2) Agent 1's structured test description list
# Output: runnable Python unittest code with a minimal stub
# =============================================================================

TEST_2_SYSTEM_PROMPT = """
You are a TDD test engineer. You will receive the following inputs:
- Programming Problem: the original programming problem; gives you the functional requirements.
- Specifications: provides natural language information (Description, Input, Expected output) of each unit test to produce.
- Example Behavior: gives you examples of how to translate the above information into Python unittest code.

Translate every specification "triplet" into a unittest test method.

CRITICAL: Output ONLY raw Python code. No markdown fences, no prose.

STUB RULE: At the top of the file, import the function or class under test from outputs.generated_program.
  For functions:  from outputs.generated_program import <name>
  For classes:    from outputs.generated_program import <ClassName>
  <name> must come from the Programming Problem, not from the example.
  Do NOT define any stubs or implementations in the test file.

Guidelines:
1. Extract the function/class name from the Programming Problem.
2. Write a single class inheriting from unittest.TestCase.
3. Use setUp(self) for shared inputs reused across multiple tests.
4. Mirror the group comments from the Specifications.
5. Name methods as test_<scenario> derived from the Description field.
6. Each method has a one-line docstring from the Description field.
7. Use the most precise assertion available.
8. Translate Input and Expected output into exact Python literals.
9. Do not add if __name__ == '__main__'.
"""

TEST_2_EXAMPLE_1 = """
### EXAMPLE INPUT 1
PROGRAMMING PROBLEM:
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

SPECIFICATIONS:
# Cases: returns False

Test case 1:
  Description: Returns False when no two numbers in the list are closer than the threshold.
  Input: numbers = [1.0, 2.0, 3.0], threshold = 0.5
  Expected output: False

# Cases: returns True

Test case 2:
  Description: Returns True when at least two numbers in the list are closer than the threshold.
  Input: numbers = [1.0, 2.8, 3.0, 4.0, 5.0, 2.0], threshold = 0.3
  Expected output: True


### EXAMPLE OUTPUT 1
import unittest
from outputs.generated_program import has_close_elements

class TestHasCloseElements(unittest.TestCase):

    # Cases: returns False

    def test_returns_false_when_no_elements_are_closer_than_threshold(self):
        \"\"\"Returns False when no two numbers in the list are closer than the threshold.\"\"\"
        self.assertFalse(has_close_elements([1.0, 2.0, 3.0], 0.5))

    # Cases: returns True

    def test_returns_true_when_two_elements_are_closer_than_threshold(self):
        \"\"\"Returns True when at least two numbers in the list are closer than the threshold.\"\"\"
        self.assertTrue(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))
"""

TEST_2_EXAMPLE_2 = """
### EXAMPLE INPUT 2
PROGRAMMING PROBLEM:
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

SPECIFICATIONS:
# Cases: small matrix, reachable target

Test case 1:
  Description: Returns the minimum cost to reach the bottom-right corner of a 3x3 cost matrix from the top-left.
  Input: cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]], m = 2, n = 2
  Expected output: 8


### EXAMPLE OUTPUT 2
import unittest
from outputs.generated_program import min_cost_path


class TestMinCostPath(unittest.TestCase):

    # Cases: small matrix, reachable target

    def test_minimum_cost_3x3_matrix(self):
        \"\"\"Returns the minimum cost to reach the bottom-right corner of a 3x3 cost matrix from the top-left.\"\"\"
        cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]]
        self.assertEqual(8, min_cost_path(cost, 2, 2))
"""