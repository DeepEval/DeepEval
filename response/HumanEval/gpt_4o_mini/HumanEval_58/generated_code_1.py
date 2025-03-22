def method(list1, list2):
    # Convert both lists to sets to find unique elements
    set1 = set(list1)
    set2 = set(list2)
    
    # Find the intersection of both sets
    common_elements = set1.intersection(set2)
    
    # Convert the result back to a sorted list
    output = sorted(common_elements)
    
    return output

# Test case
list_a = [1, 2, 3, 4, 5, 5]
list_b = [4, 5, 6, 7, 8]

# Expected output: [4, 5]
print(method(list_a, list_b))  # This should print [4, 5]