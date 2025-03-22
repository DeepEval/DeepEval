from collections import Counter

def method(nums):
    # Count the frequency of each integer in the list
    frequency = Counter(nums)
    
    # Initialize a variable to track the greatest integer that meets the criteria
    max_integer = -1
    
    # Iterate through the frequency dictionary
    for num, freq in frequency.items():
        # Check if the frequency is greater than or equal to the integer itself
        if freq >= num:
            max_integer = max(max_integer, num)
    
    return max_integer

# Test case for validation
if __name__ == "__main__":
    test_list = [3, 3, 2, 2, 1, 1, 1]
    output = method(test_list)
    print(output)  # Should print 3 as it appears twice (>= 3)