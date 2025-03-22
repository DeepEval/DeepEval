def method():
    arr = [1, 2, 3, 4, 5]
    sum_first_last = arr[0] + arr[-1]
    if sum_first_last % 2 == 0:  
        arr.sort(reverse=True)
    else:
        arr.sort()
    output = arr.copy()  # Return a copy to avoid modifying the original
    return output 

# Test Case
result = method()
print(f"Original array: {arr}")  
print(f"Sorted array: {result}")