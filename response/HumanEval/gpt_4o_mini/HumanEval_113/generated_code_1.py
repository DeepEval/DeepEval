def method():
    # Example input list of strings containing digits
    input_list = ["123", "456", "7890", "13579"]
    
    # Create an output list to store the counts of odd digits
    output = []

    # Iterate through each string in the input list
    for s in input_list:
        # Count the number of odd digits in the current string
        odd_count = sum(1 for char in s if char in '13579')
        # Append the count to the output list
        output.append(odd_count)

    return output

# Test case for validation
if __name__ == "__main__":
    result = method()
    print(result)  # Expected output: [2, 0, 4, 5]