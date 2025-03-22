def method():
    input_list = ["apple", "banana", "cherry", "date"]
    if not input_list:
        return None
    longest_string = input_list[0]
    for string in input_list:
        if len(string) > len(longest_string):
            longest_string = string
        elif len(string) == len(longest_string) and input_list.index(string) < input_list.index(longest_string):
            longest_string = string
    return longest_string

# Test case
print(method())  # Output should be "banana"