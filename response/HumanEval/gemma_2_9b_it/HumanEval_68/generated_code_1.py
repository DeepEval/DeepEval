def method(tree):
    smallest_even = float('inf')
    smallest_index = -1

    for i, node in enumerate(tree):
        if node % 2 == 0 and node < smallest_even:
            smallest_even = node
            smallest_index = i
    
    if smallest_even == float('inf'):
      return []
    return [smallest_even, smallest_index]

# Test case
test_tree = [1, 4, 3, 2, 5, 6]
output = method(test_tree)
print(output)  # Output: [2, 3]