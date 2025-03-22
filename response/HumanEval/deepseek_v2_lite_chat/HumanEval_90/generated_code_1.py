def method():
    def next_smallest(lst):
        if not lst:
            return None
        first, second = float('inf'), float('inf')
        for num in lst:
            if num < first:
                second = first
                first = num
            elif num < second and num != first:
                second = num
        return second if second != float('inf') else None
    return next_smallest

# Test cases
def test_next_smallest():
    # assert method()()([1, 2, 3, 4, 5]) == 2
    # assert method()([5, 1, 4, 3, 2]) == 2
    # assert method()([1, 1]) == None
    # assert method()([100]) == None
    # assert method()([]) == None
    print(method()([1,1]))

test_next_smallest()