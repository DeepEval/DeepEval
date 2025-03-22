def method():
    def count_starting_with_1(n):
        if n == 1:
            return 1
        return 9 * (10 ** (n - 2))

    def count_ending_with_1(n):
        if n == 1:
            return 1
        return 9 * (10 ** (n - 2))

    def count_both_starting_and_ending_with_1(n):
        if n == 1:
            return 1
        return 10 ** (n - 2)

    n = 3  # Example input
    count_start_with_1 = count_starting_with_1(n)
    count_end_with_1 = count_ending_with_1(n)
    count_both = count_both_starting_and_ending_with_1(n)

    output = count_start_with_1 + count_end_with_1 - count_both
    return output

# Test case
print(method())  # Example test case