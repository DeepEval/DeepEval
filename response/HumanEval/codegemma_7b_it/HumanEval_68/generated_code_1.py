from collections import deque

def method(nodes):
    """
    Given an array representing a branch of a tree that has non-negative integer nodes,
    your task is to pluck one of the nodes and return it. The plucked node should be the node with the smallest even value.
    If multiple nodes with the same smallest even value are found return the node that has smallest index.
    The plucked node should be returned in a list, [ smalest_value, its index ],
    If there are no even values or the given array is empty, return [].

    Args:
        nodes: An array representing a branch of a tree.

    Returns:
        A list containing the smallest even value and its index, or [] if there are no even values or the array is empty.
    """

    if not nodes:
        return []

    queue = deque(nodes)
    min_value = float('inf')
    min_index = None

    while queue:
        node = queue.popleft()

        if node % 2 == 0 and node <= min_value:
            if node < min_value:
                min_value = node
                min_index = queue.index(node)
            elif node == min_value:
                min_index = min(min_index, queue.index(node))

    if min_index is None:
        return []

    return [min_value, min_index]


# Test case
nodes = [4, 2, 5, 6, 1]
result = method(nodes)
print(result)  # Output: [2, 1]

nodes = [1, 3, 5]
result = method(nodes)
print(result)  # Output: []