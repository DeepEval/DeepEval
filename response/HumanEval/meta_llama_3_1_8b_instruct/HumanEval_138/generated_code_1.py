def method(n):
    """
    This function checks whether a given number can be expressed as the sum of exactly 4 positive even numbers.

    Args:
    n (int): The number to check.

    Returns:
    bool: True if the number can be expressed as the sum of exactly 4 positive even numbers, False otherwise.
    """

    def is_sum_of_four_even_numbers(i, target, numbers):
        # Base case: If the target becomes zero, it means we've found a combination of four even numbers that sum up to the target.
        if target == 0:
            return True
        
        # If the target becomes negative, it means the current combination of numbers exceeds the target, so we return False.
        if target < 0:
            return False
        
        # Initialize a counter to keep track of the number of even numbers in the current combination.
        even_count = 0
        
        # Iterate over all numbers from i to n.
        for j in range(i, n + 1):
            # Check if the current number is even.
            if j % 2 == 0:
                # Increment the even count.
                even_count += 1
                
                # If the even count is greater than 4, we can break the loop because the remaining numbers will not make the even count greater than 4.
                if even_count > 4:
                    break
                
                # Recursively call the function with the updated target and the current number.
                if is_sum_of_four_even_numbers(j + 1, target - j, numbers + [j]):
                    return True
        
        # If we've iterated over all numbers and haven't found a combination that sums up to the target, we return False.
        return False

    # Initialize the number of even numbers in the combination to 0.
    even_count = 0
    
    # Iterate over all numbers from 1 to n.
    for i in range(1, n + 1):
        # Check if the current number is even.
        if i % 2 == 0:
            # Increment the even count.
            even_count += 1
            
            # If the even count is greater than 4, we can break the loop because the remaining numbers will not make the even count greater than 4.
            if even_count > 4:
                break
            
            # Recursively call the function with the updated target and the current number.
            if is_sum_of_four_even_numbers(i + 1, n - i, [i]):
                return True
    
    # If we've iterated over all numbers and haven't found a combination that sums up to the target, we return False.
    return False

# Test case
n = 12
output = method(n)
print(f"Can {n} be expressed as the sum of exactly 4 positive even numbers? {output}")