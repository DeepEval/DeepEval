from collections import defaultdict

def method(input_list):
    seen_elements = defaultdict(int)
    output = []

    for element in input_list:
        if seen_elements[element] == 0:
            output.append(element)
        seen_elements[element] += 1

    return output

# Test Case
input_list = [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 7]
output = method(input_list)
print(output)  # Output: [1, 2, 3, 4, 5, 6, 7]