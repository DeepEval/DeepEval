from collections import Counter
import re

def method():
    # Get the input string from the user
    # input_string = input("Enter a space separated lowercase letters: ")
    input_string = "hello world"

    # Remove any spaces from the input string
    input_string = re.sub(r"\s+", "", input_string)

    # Convert the input string to a list of letters
    letters = list(input_string)

    # Create a dictionary of letter counts
    letter_counts = Counter(letters)

    # Get the most common letter and its count
    most_common_letter = letter_counts.most_common(1)[0][0]
    most_common_count = letter_counts.most_common(1)[0][1]

    # Create a dictionary of letters with the most repetition
    output = {letter: count for letter, count in letter_counts.items() if count == most_common_count}

    # Return the dictionary of letters with the most repetition
    return output

# Test case
input_string = "hello world"
output = method()
print(output)