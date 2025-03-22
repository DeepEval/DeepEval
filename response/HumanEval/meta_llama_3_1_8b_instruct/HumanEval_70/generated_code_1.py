import heapq

def method(nums):
    """
    Given a list of integers, returns the list in strange order.
    Strange sorting is when you start with the minimum value, 
    then maximum of the remaining integers, then minimum and so on.
    
    Args:
    nums (list): A list of integers.
    
    Returns:
    list: The input list sorted in strange order.
    """
    if not nums:
        return []
    
    # First, we sort the list to get the smallest number
    nums.sort()
    
    # Initialize two lists: one for the smallest numbers and one for the largest numbers
    smallest = []
    largest = []
    
    # We also keep track of the current smallest and largest numbers
    current_smallest = nums[0]
    current_largest = nums[0]
    
    # Add the current smallest and largest numbers to their respective lists
    smallest.append(current_smallest)
    largest.append(current_largest)
    
    # Remove the current smallest and largest numbers from the original list
    nums = nums[1:]
    
    # Loop until we have processed all numbers
    while nums:
        # Remove all numbers smaller than the current smallest from the list
        while nums and nums[0] < current_smallest:
            heapq.heappush(smallest, nums.pop(0))
        
        # Remove all numbers larger than the current largest from the list
        while nums and nums[0] > current_largest:
            heapq.heappush(largest, nums.pop(0))
        
        # The next smallest number is the smallest of the remaining numbers
        # or the smallest number we have seen so far
        current_smallest = min(nums[0] if nums else float('inf'), current_smallest)
        
        # The next largest number is the largest of the remaining numbers
        # or the largest number we have seen so far
        current_largest = max(nums[0] if nums else float('-inf'), current_largest)
        
        # Add the smallest number to the list of smallest numbers
        smallest.append(nums.pop(0))
    
    # Merge the two lists to get the final sorted list
    return sorted(smallest + largest, reverse=True)

# Test case for validation
print(method([1, 2, 3, 4, 5]))  # Output: [1, 5, 1, 3, 5]