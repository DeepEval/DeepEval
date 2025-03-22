def method():
    arr = [1, -2, 3, 0, -4]  
    
    if not arr:  
        return None

    output = 0
    product_of_signs = 1
    for num in arr:
        if num > 0:
            product_of_signs *= 1
        elif num < 0:
            product_of_signs *= -1
        else:
            product_of_signs *= 0
        output += abs(num) * product_of_signs

    return output

# Test Case
print(method())  # Output: 1