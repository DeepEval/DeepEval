def method(s, n):
  """
  Finds all words in a string containing a specific number of consonants.

  Args:
    s: The input string.
    n: The target number of consonants.

  Returns:
    A list of words from the string that contain exactly n consonants,
    in the order they appear in the string.
  """
  vowels = "aeiouAEIOU"
  words = s.split()
  output = []
  for word in words:
    consonant_count = 0
    for letter in word:
      if letter not in vowels:
        consonant_count += 1
    if consonant_count == n:
      output.append(word)
  return output

# Test Case
s = "This is a test string with some words"
n = 3
result = method(s, n)
print(f"Words with {n} consonants: {result}")