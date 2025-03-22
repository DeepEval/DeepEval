from collections import Counter

def method():
    def find_greatest_integer(numbers):
        # Count the frequency of each number in the list
        freq = Counter(numbers)
        # Initialize the maximum value and the greatest integer
        max_value = -1
        greatest_integer = -1
        # Iterate over the frequencies
        for number, count in freq.items():
            # Check if the frequency is greater than or equal to the number itself
            if count >= number:
                # Update the maximum value and the greatest integer
                max_value = max(max_value, number)
                greatest_integer = max(greatest_integer, number)
        return greatest_integer

    # Example list of positive integers
    numbers = [1, 2, 3, 4, 2, 2, 3, 3, 3, 4, 4]
    # Find and return the greatest integer
    return find_greatest_integer(numbers)

# Test case
print(method())  # Should return 4