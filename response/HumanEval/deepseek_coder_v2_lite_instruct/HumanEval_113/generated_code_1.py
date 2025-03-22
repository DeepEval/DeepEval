def method():
    # Example input list of strings
    input_list = ["123", "456", "789"]
    
    # Function to count the number of odd digits in a string
    def count_odd_digits(s):
        return sum(1 for char in s if int(char) % 2 != 0)
    
    # Create the output list
    output = [str(count_odd_digits(s)) for s in input_list]
    
    return output

# Test case
output = method()
print(output)  # Expected output: ["2", "2", "3"]