import collections

def method(arr):
    if not arr:
        return True
    if len(arr) == 1:
        return True

    count = collections.Counter(arr)
    max_count = max(count.values())

    if max_count == len(arr):
        return True

    if max_count == len(arr) - 1:
        return True

    if max_count > len(arr) - max_count:
        return False

    return True

# Test case
arr = [4, 3, 2, 1]
print(method(arr))  # Output: True

arr = [5, 4, 3, 2, 1]
print(method(arr))  # Output: False