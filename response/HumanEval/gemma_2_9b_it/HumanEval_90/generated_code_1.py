def method():
    data = [1, 2, 3, 4, 5]  
    # your code here

    output = next_smallest(data) 
    return output

def next_smallest(nums):
    if len(nums) < 2:
        return None
    
    # Find the smallest and second smallest elements
    smallest = min(nums)
    second_smallest = float('inf')  # Initialize to a very large number
    for num in nums:
        if num != smallest and num < second_smallest:
            second_smallest = num
    return second_smallest

# Test Case
print(method())  # Expected output: 2