def method():
    # Sample input list
    input_list = [1, 2.5, "hello", 3, "world", 4.0, 5]
    
    # Filter the list to include only integers
    output = [item for item in input_list if isinstance(item, int)]
    
    return output

# Test case
if __name__ == "__main__":
    result = method()
    print("Filtered integers:", result)  # Expected output: [1, 3, 5]