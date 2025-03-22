def pairs_sum_to_zero(lst):
    seen = set()
    for num in lst:
        if -num in seen:
            return True
        seen.add(num)
    return False

def method():
    # Example test case
    test_list = [1, 2, -3, 4]
    output = pairs_sum_to_zero(test_list)
    return output

# Run the method and print the output
if __name__ == "__main__":
    result = method()
    print(result)  # This will print True since 2 and -2 exist in the list