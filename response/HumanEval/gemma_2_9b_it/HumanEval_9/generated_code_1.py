def method():
    data = [10, 5, 20, 15, 25, 18, 30]
    k = 3  # Window size
    output = []
    for i in range(len(data)):
        if i < k:
            output.append(data[i])
        else:
            max_element = max(data[i-k+1:i+1])
            output.append(max_element)
    return output

# Test case
print(method())  # Output: [10, 10, 20, 20, 25, 25, 30]