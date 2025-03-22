from collections import Counter

def method(input_list):
    # Count occurrences of each element in the list
    counts = Counter(input_list)
    
    # Create a new list with elements that occur exactly once, preserving order
    output = [num for num in input_list if counts[num] == 1]
    
    return output

# Test case for validation
if __name__ == "__main__":
    test_input = [4, 5, 6, 4, 7, 8, 5, 9]
    output = method(test_input)
    print(output)  # Expected output: [6, 7, 8, 9]