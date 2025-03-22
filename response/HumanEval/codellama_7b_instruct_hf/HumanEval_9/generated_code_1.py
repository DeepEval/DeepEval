import numpy as np

def method():
    input_list = [10, 5, 15, 20, 8, 12, 4]
    output = []
    for i in range(len(input_list)):
        output.append(max(input_list[:i+1]))
    return output

# Test case
input_list = [10, 5, 15, 20, 8, 12, 4]
output = method()
print(output)