import math

def method(numbers):
    # Round each number up to the nearest integer and square it
    squared_sum = sum(math.ceil(num) ** 2 for num in numbers)
    return squared_sum

# Test case for validation
if __name__ == "__main__":
    test_numbers = [1.2, 2.3, 3.4, 4.5]  # Example input
    output = method(test_numbers)
    print(f"The sum of squared numbers is: {output}")