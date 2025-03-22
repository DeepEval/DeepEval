def method(nums):
    # Initialize the minimum sum with the first element
    min_sum = float('inf')
    current_sum = 0
    
    for num in nums:
        current_sum += num  # Add the current number to the current sum
        if current_sum < min_sum:
            min_sum = current_sum  # Update minimum sum if current sum is lower
        
        if current_sum > 0:
            current_sum = 0  # Reset current sum if it becomes positive

    return min_sum

# Test case for validation
if __name__ == "__main__":
    test_array = [3, -2, 5, -1, 2, -3]
    output = method(test_array)
    print("The minimum sum of any non-empty sub-array is:", output)  # Expected output: -4