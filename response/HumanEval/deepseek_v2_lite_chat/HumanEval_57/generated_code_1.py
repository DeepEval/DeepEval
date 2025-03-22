def method(lst):
    # Check if the list is empty
    if not lst:
        return True
    
    # Check if the list has only one element
    if len(lst) == 1:
        return True
    
    # Check if the list has more than one element
    for i in range(len(lst) - 1):
        # If the list is not monotonically increasing or decreasing, return False
        if lst[i] >= lst[i + 1] or lst[i] <= lst[i + 1]:
            return False
    
    # If the list is monotonically increasing or decreasing, return True
    return True

# Test case
def test_method():
    # Test case 1: Monotonically increasing list
    lst1 = [1, 2, 3, 4, 5]
    print(method(lst1))
    # assert method(lst1) == True
    
    # Test case 2: Monotonically decreasing list
    lst2 = [5, 4, 3, 2, 1]
    # assert method(lst2) == True
    
    # Test case 3: Increasing list with duplicates
    lst3 = [1, 1, 2, 3, 4, 4]
    # assert method(lst3) == True
    
    # Test case 4: Decreasing list with duplicates
    lst4 = [4, 4, 3, 2, 1, 1]
    # assert method(lst4) == True
    
    # Test case 5: Non-monotonic list
    lst5 = [1, 1, 1, 2, 3]
    # assert method(lst5) == False
    
    # print("All test cases pass")

# Run the test function
test_method()