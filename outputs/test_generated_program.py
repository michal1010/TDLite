from outputs.generated_program import has_close_elements

class TestHasCloseElements(unittest.TestCase):

    def test_returns_false_when_no_elements_are_closer_than_threshold(self):
        self.assertFalse(has_close_elements([1.0, 2.0, 3.0], 0.5))

    def test_returns_true_when_two_elements_are_closer_than_threshold(self):
        self.assertTrue(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))
