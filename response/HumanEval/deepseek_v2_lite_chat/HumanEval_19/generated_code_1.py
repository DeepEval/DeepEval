def method():
    # Define a dictionary mapping the string representations of numbers to their numerical values
    num_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    
    # Get the input string
    # input_string = input("Enter a space-delimited string of numbers (zero to nine): ")
    input_string = "three seven one four six two nine zero eight five"
    
    # Split the input string into a list of words
    words = input_string.split()
    
    # Convert each word to its numerical value and sort the list
    sorted_words = sorted(words, key=lambda word: num_dict[word])
    
    # Join the sorted list into a string and return it
    return ' '.join(sorted_words)

# Example test case
test_case = "three seven one four six two nine zero eight five"
print("Test Case:", test_case)
sorted_output = method()
print("Sorted Output:", sorted_output)