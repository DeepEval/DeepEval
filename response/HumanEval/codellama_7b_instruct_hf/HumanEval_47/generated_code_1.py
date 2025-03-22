from statistics import median

def method(l):
    """
    This method returns the median of elements in the list l.
    """
    output = median(l)
    return output


# Test case for validation
l = [1, 2, 3, 4, 5]
print(method(l))