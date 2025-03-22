def method(arr):
    if not arr:  # Return None for an empty array
        return None
    
    sum_of_magnitudes = sum(abs(num) for num in arr)
    
    # Calculate the product of signs
    product_of_signs = 1
    for num in arr:
        if num > 0:
            sign = 1
        elif num < 0:
            sign = -1
        else:
            sign = 0
        
        product_of_signs *= sign
        if product_of_signs == 0:  # Early exit if product becomes zero
            break

    output = sum_of_magnitudes * product_of_signs
    return output

# Test case
test_case = [1, -2, 3, 0, -4]
print(method(test_case))  # Expected output: 0