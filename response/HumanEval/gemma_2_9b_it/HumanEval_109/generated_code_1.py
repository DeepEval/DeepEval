def method():
    arr = [3, 1, 2] 
    n = len(arr)
    sorted_arr = sorted(arr)  
    
    for i in range(n):
        if arr[i] != sorted_arr[i]:
            # Check if rotating the array would fix the mismatch
            if arr == sorted_arr:
                return True
            else:
                arr.append(arr.pop(0)) 

    return False

output = method()
print(output)