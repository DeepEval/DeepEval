def method():
    def is_even(n):
        """Check if a number is even."""
        return n % 2 == 0

    def find_biggest_even(x, y):
        """Find the biggest even number in the range [x, y] inclusive."""
        # Check if x or y is not an integer
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("x and y must be integers.")

        # Check if x and y are positive
        if x <= 0 or y <= 0:
            raise ValueError("x and y must be positive.")

        # Find the biggest even number in the range [x, y] inclusive
        even_nums = [n for n in range(x, y + 1) if is_even(n)]
        if even_nums:
            return max(even_nums)
        else:
            return -1

    try:
        # x = int(input("Enter the first positive number: "))
        # y = int(input("Enter the second positive number: "))

        # 修改：
        x = 1
        y = 10
        
        output = find_biggest_even(x, y)
        return output
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Test case
print(method())