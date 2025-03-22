def method(input_list):
    """
    This function takes a list of strings as input where each string consists of only digits.
    It returns a list where each element i is replaced by the number of odd digits in the i'th string of the input.

    Args:
        input_list (list): A list of strings where each string consists of only digits.

    Returns:
        list: A list where each element i is replaced by the number of odd digits in the i'th string of the input.
    """
    output = []
    for string in input_list:
        # Count the number of odd digits in the string
        odd_count = sum(1 for char in string if int(char) % 2!= 0)
        output.append(f"The number of odd elements in the string is: {odd_count}")
    return output


# Test case
input_list = ["123", "456", "789", "1011"]
print(method(input_list))