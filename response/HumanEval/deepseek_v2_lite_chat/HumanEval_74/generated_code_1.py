import collections

def method(list1, list2):
    # Count the number of characters in each string in the first list
    count1 = sum(len(s) for s in list1)
    
    # Count the number of characters in each string in the second list
    count2 = sum(len(s) for s in list2)
    
    # Determine the list with the fewer or equal number of characters
    if count1 <= count2:
        return list1
    else:
        return list2

# Test case
list1 = ["hello", "world", "python", "programming"]
list2 = ["AI", "developer", "coding", "python", "programming"]

print(method(list1, list2))  # Output should be ['AI', 'developer', 'coding']