import itertools

def method():
    # Sample list of numbers for testing
    lst1 = [1, 3, 5, 7, 9]
    lst2 = [2, 4, 6, 8, 10]
    
    # Calculate the sum of elements in each list
    sum1 = sum(lst1)
    sum2 = sum(lst2)
    
    # Check if it's possible to make lst1 an even list by exchanging elements
    if sum1 <= sum2:
        return "YES"
    else:
        return "NO"

# Test case
output = method()
print(output)