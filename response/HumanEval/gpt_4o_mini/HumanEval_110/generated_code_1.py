def method(lst1, lst2):
    # Check if lst1 has only even numbers
    if all(x % 2 == 0 for x in lst1):
        return "YES"
    
    # Check if lst2 has at least one even number
    if any(x % 2 == 0 for x in lst2):
        return "YES"
    
    # If lst1 has odd numbers and lst2 has no even numbers, return "NO"
    return "NO"

# Test case
lst1 = [1, 3, 5]  # odd numbers
lst2 = [2, 4, 6]  # even numbers
output = method(lst1, lst2)  # Should return "YES" since we can exchange odd numbers in lst1 with even numbers in lst2
print(output)  # Output should be "YES"

# Another test case where it should return "NO"
lst1 = [1, 3, 5]  # odd numbers
lst2 = [1, 3, 5]  # odd numbers
output = method(lst1, lst2)  # Should return "NO" since there are no even numbers to exchange
print(output)  # Output should be "NO"