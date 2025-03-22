def method(list1, list2):
    # Convert lists to sets to find common elements and ensure uniqueness
    set1 = set(list1)
    set2 = set(list2)
    
    # Find the intersection of the two sets to get common elements
    common_elements = set1.intersection(set2)
    
    # Convert the set of common elements back to a sorted list
    sorted_common_elements = sorted(list(common_elements))
    
    return sorted_common_elements

# Test case
list1 = [1, 2, 2, 3, 4]
list2 = [3, 4, 4, 5, 6]
output = method(list1, list2)
print(output)  # Expected output: [3, 4]