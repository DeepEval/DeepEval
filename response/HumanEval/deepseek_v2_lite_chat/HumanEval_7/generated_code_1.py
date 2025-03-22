def method(input_list, substring):
    # Function to check if a string contains the given substring
    def contains_substring(string):
        return substring in string
    
    # Filter the input list to include only those strings that contain the substring
    output = list(filter(contains_substring, input_list))
    
    return output