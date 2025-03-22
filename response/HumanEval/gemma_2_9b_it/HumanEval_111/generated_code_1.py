from collections import Counter

def method(text):
  """
  Given a string representing a space separated lowercase letters, return a dictionary
  of the letter with the most repetition and containing the corresponding count.
  If several letters have the same occurrence, return all of them. 

  Args:
      text: A string of lowercase letters separated by spaces.

  Returns:
      A dictionary where the keys are the most repeated letters and the values are their counts.
  """
  letter_counts = Counter(text.split())
  max_count = max(letter_counts.values())
  output = {letter: count for letter, count in letter_counts.items() if count == max_count}
  return output

# Test case
test_string = "this is a test string test"
output = method(test_string)
print(f"Input: {test_string}")
print(f"Output: {output}")