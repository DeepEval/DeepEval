def method():
    lst1 = [1, 3, 5, 7]
    lst2 = [2, 4, 6, 8]
    
    odd_in_lst1 = [num for num in lst1 if num % 2 != 0]
    even_in_lst2 = [num for num in lst2 if num % 2 == 0]
    
    if len(odd_in_lst1) <= len(even_in_lst2):
        return "YES"
    else:
        return "NO"

# Test case
output = method()
print(output)  # Expected output: "YES"