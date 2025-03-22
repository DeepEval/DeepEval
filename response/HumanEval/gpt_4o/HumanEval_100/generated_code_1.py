def method(n):
    levels = [n]
    current_stones = n
    
    for _ in range(1, n):
        if current_stones % 2 == 0:
            # If even, increment to the next even number
            current_stones += 2
        else:
            # If odd, increment to the next odd number
            current_stones += 2
        
        levels.append(current_stones)
        
    return levels

# Test case
print(method(5))  # For example, with n = 5, the output should be [5, 7, 9, 11, 13]