def method(n):
    stones = []
    current_stones = n
    
    for i in range(n):
        stones.append(current_stones)
        if n % 2 == 0:  # n is even
            current_stones += 2  # Next even number
        else:  # n is odd
            current_stones += 2  # Next odd number
    
    return stones

# Test case for validation
test_n = 5  # Example input
output = method(test_n)
print(f'Number of stones in each level for n = {test_n}: {output}')