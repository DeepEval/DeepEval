def method(n):
    count_of_sevens = 0
    
    for i in range(1, n):
        if i % 11 == 0 or i % 13 == 0:
            count_of_sevens += str(i).count('7')
    
    return count_of_sevens

# Test case
n = 100
output = method(n)
print(output)  # Should print the number of '7's in the required range