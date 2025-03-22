def method():
    arr = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    n = len(arr)
    for i in range(1, n):
        if arr[i] < arr[i - 1]:
            return i - 1 
    return -1

output = method()
print(f"Output: {output}")