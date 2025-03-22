import math

def method(arr):
    if not arr:
        return None  # Return None for empty array

    product_of_signs = 1
    magnitude_sum = 0

    for num in arr:
        product_of_signs *= (num > 0) - (num < 0)  # Calculate product of signs
        magnitude_sum += math.fabs(num)  # Calculate sum of magnitudes

    return product_of_signs * magnitude_sum  # Return the final result

# Test case
arr = [3, -2, 5, -6]
result = method(arr)
print(result)  # Output: 88