def method(input_string):
    """
    This function takes a space-delimited string of numbers from 'zero' to 'nine',
    sorts them from smallest to largest and returns the sorted string.

    Parameters:
    input_string (str): A space-delimited string of numbers.

    Returns:
    str: The input string with numbers sorted from smallest to largest.
    """
    # Define a dictionary to map number names to their integer values
    num_dict = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5,'six': 6,'seven': 7, 'eight': 8, 'nine': 9
    }

    # Split the input string into a list of numbers
    num_list = input_string.split()

    # Convert the list of number names to a list of integers
    num_list = [num_dict[num] for num in num_list]

    # Sort the list of integers in ascending order
    num_list.sort()

    # Convert the list of integers back to number names and join them into a string
    output =''.join([key for key, value in num_dict.items() if value == num] for num in num_list)

    return output


# Test case for validation
input_string = "one two three four five six seven eight nine"
output = method(input_string)
print(output)  # Expected output: "zero one two three four five six seven eight nine"