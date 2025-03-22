from collections import defaultdict

def method():
    list1 = [1, 2, 3, 4, 5, 3, 2]
    list2 = [2, 4, 6, 8, 10, 2, 4]

    # Create a dictionary to store the frequency of elements in list1
    freq = defaultdict(int)
    for element in list1:
        freq[element] += 1

    # Create a list to store the sorted unique common elements
    common_elements = []
    for element in list2:
        if element in freq and freq[element] > 0:
            common_elements.append(element)
            freq[element] -= 1

    # Sort the common elements in ascending order
    common_elements.sort()

    return common_elements

# Test case
# assert method() == [2, 4]

method()