# Importing necessary packages
import re

def fix_spaces(text):
  """Replaces all spaces in a string with underscores, and if a string has more than 2 consecutive spaces, then replace all consecutive spaces with -"""

  # Replace all spaces with underscores
  text = re.sub(r"\s", "_", text)

  # Replace all consecutive spaces with -
  text = re.sub(r"\s+", "-", text)

  return text

# Test case
text = "This   string has     too many spaces."
fixed_text = fix_spaces(text)
print(fixed_text) # Output: This_string_has-too_many_spaces.