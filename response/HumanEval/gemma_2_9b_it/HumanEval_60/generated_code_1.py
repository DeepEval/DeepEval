def method():
    """
    sum_to_n is a function that sums numbers from 1 to n. 
    """
    n = 5  # Example input, can be changed
    output = sum(range(1, n + 1))  
    return output

# Test case
result = method()
print(f"The sum of numbers from 1 to 5 is: {result}")