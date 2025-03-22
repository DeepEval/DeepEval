def method(numbers):
    # Check if the input list is empty
    if not numbers:
        return (0, 1)
    
    # Initialize sum and product
    total_sum = 0
    total_product = 1
    
    # Calculate sum and product
    for num in numbers:
        total_sum += num
        total_product *= num
    
    return (total_sum, total_product)

# Test case
if __name__ == "__main__":
    # Example test case
    numbers = [1, 2, 3, 4]
    output = method(numbers)
    print(f"Input: {numbers} => Output: {output}")  # Output should be (10, 24)

    # Test case for empty list
    empty_numbers = []
    empty_output = method(empty_numbers)
    print(f"Input: {empty_numbers} => Output: {empty_output}")  # Output should be (0, 1)