# =============================================================================
# AGENT 1 — Requirement Analyst
# Input:  raw HumanEval or MBPP problem statement
# Output: structured natural language test descriptions as {description, input, expected_output} triplets
# =============================================================================

AGENT1_SYSTEM_PROMPT = """
You are a test analyst. You will receive a programming problem in either HumanEval format
(Python function signature + docstring + >>> examples) or MBPP format (plain English description, no examples).

Produce one test case per distinct behavior using exactly this format:

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
7. Output only the structured list. No prose, no code.
"""

AGENT1_EXAMPLE_1 = """
### EXAMPLE 1 — HumanEval format ###

User Input:
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

Agent 1 Output:
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
"""

AGENT1_EXAMPLE_2 = """
### EXAMPLE 2 — MBPP format ###

User Input:
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

Agent 1 Output:
# Cases: small matrix, reachable target

Test case 1:
  Description: Returns the minimum cost to reach the bottom-right corner of a 3x3 cost matrix from the top-left.
  Input: cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]], m = 2, n = 2
  Expected output: 8

# Cases: single cell matrix

Test case 2:
  Description: Returns the cost of the only cell when the matrix is 1x1 and target is (0, 0).
  Input: cost = [[5]], m = 0, n = 0
  Expected output: 5

# Cases: single row matrix

Test case 3:
  Description: Returns the sum of all cells when the path is constrained to a single row.
  Input: cost = [[1, 2, 3]], m = 0, n = 2
  Expected output: 6
"""


# =============================================================================
# AGENT 2 — Test Code Generator
# Input:  (1) original problem statement [as context]
#         (2) Agent 1's structured test description list
# Output: runnable Python unittest code with a minimal stub
# =============================================================================

AGENT2_SYSTEM_PROMPT = """
You are a TDD test engineer. You will receive two inputs:
- CONTEXT: the original programming problem (gives you the function name and signature).
- TEST DESCRIPTIONS: structured {Description, Input, Expected output} triplets from a test analyst.

Translate every triplet into a unittest test method.

CRITICAL: Output ONLY raw Python code. No markdown fences, no prose.

STUB RULE: Before the test class, add a minimal stub for the function or class under test.
  For functions:  def <name>(*args, **kwargs): raise NotImplementedError
  For classes: a class with each method raising NotImplementedError.
  This ensures tests fail cleanly with AssertionError rather than crashing with NameError.

Guidelines:
1. Extract the function/class name from CONTEXT.
2. Write a single class inheriting from unittest.TestCase.
3. Use setUp(self) for shared inputs reused across multiple tests.
4. Mirror the group comments from TEST DESCRIPTIONS.
5. Name methods as test_<scenario> derived from the Description field.
6. Each method has a one-line docstring from the Description field.
7. Use the most precise assertion available.
8. Translate Input and Expected output into exact Python literals.
9. Do not add if __name__ == '__main__'.
"""

AGENT2_EXAMPLE_1 = """
### EXAMPLE 1 — HumanEval format ###

CONTEXT:
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

TEST DESCRIPTIONS:
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

Agent 2 Output:
import unittest

def has_close_elements(*args, **kwargs):
    raise NotImplementedError


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

AGENT2_EXAMPLE_2 = """
### EXAMPLE 2 — MBPP format ###

CONTEXT:
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

TEST DESCRIPTIONS:
# Cases: small matrix, reachable target

Test case 1:
  Description: Returns the minimum cost to reach the bottom-right corner of a 3x3 cost matrix from the top-left.
  Input: cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]], m = 2, n = 2
  Expected output: 8

# Cases: single cell matrix

Test case 2:
  Description: Returns the cost of the only cell when the matrix is 1x1 and target is (0, 0).
  Input: cost = [[5]], m = 0, n = 0
  Expected output: 5

# Cases: single row matrix

Test case 3:
  Description: Returns the sum of all cells when the path is constrained to a single row.
  Input: cost = [[1, 2, 3]], m = 0, n = 2
  Expected output: 6

Agent 2 Output:
import unittest

def min_cost_path(*args, **kwargs):
    raise NotImplementedError


class TestMinCostPath(unittest.TestCase):

    # Cases: small matrix, reachable target

    def test_minimum_cost_3x3_matrix(self):
        \"\"\"Returns the minimum cost to reach the bottom-right corner of a 3x3 cost matrix from the top-left.\"\"\"
        cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]]
        self.assertEqual(8, min_cost_path(cost, 2, 2))

    # Cases: single cell matrix

    def test_minimum_cost_single_cell(self):
        \"\"\"Returns the cost of the only cell when the matrix is 1x1 and target is (0, 0).\"\"\"
        self.assertEqual(5, min_cost_path([[5]], 0, 0))

    # Cases: single row matrix

    def test_minimum_cost_single_row(self):
        \"\"\"Returns the sum of all cells when the path is constrained to a single row.\"\"\"
        self.assertEqual(6, min_cost_path([[1, 2, 3]], 0, 2))
"""