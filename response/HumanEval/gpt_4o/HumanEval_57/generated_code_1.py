def method(lst):
    if not lst:
        return True  # An empty list is considered monotonically increasing and decreasing

    increasing = decreasing = True

    for i in range(1, len(lst)):
        if lst[i] > lst[i - 1]:
            decreasing = False
        if lst[i] < lst[i - 1]:
            increasing = False

    return increasing or decreasing

# Test cases for validation
print(method([1, 2, 3, 4, 5]))  # True, monotonically increasing
print(method([5, 4, 3, 2, 1]))  # True, monotonically decreasing
print(method([1, 2, 2, 3, 4]))  # True, still considered monotonically increasing
print(method([1, 3, 2]))        # False, neither increasing nor decreasing
print(method([]))               # True, empty list
print(method([1]))              # True, single element list