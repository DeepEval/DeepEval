def method(input_list, substring):
    """
    Filters a list of strings and returns a new list containing only strings that contain a given substring.

    Args:
        input_list (list): A list of strings to be filtered.
        substring (str): The substring to search for in the input list.

    Returns:
        list: A new list containing only strings that contain the given substring.
    """
    filtered_list = [string for string in input_list if substring in string]
    return filtered_list

# Example usage:
input_list = ["apple", "banana", "cherry", "date", "elderberry"]
substring = "an"
output = method(input_list, substring)
print(output)  # Output: ["banana", "cherry", "date"]

# Test case:
import unittest

class TestFilterListStrings(unittest.TestCase):
    def test_filter_list_strings(self):
        input_list = ["apple", "banana", "cherry", "date", "elderberry"]
        substring = "an"
        expected_output = ["banana", "cherry", "date"]
        self.assertEqual(method(input_list, substring), expected_output)

if __name__ == "__main__":
    unittest.main()