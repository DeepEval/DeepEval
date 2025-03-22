def method():
    import math

    def closest_integer(num_str):
        try:
            num = float(num_str)
        except ValueError:
            return "Invalid input: Please provide a valid number."

        if num > 0:
            floor_val = math.floor(num)
            ceil_val = math.ceil(num)
            if num - floor_val < ceil_val - num:
                return floor_val
            else:
                return ceil_val
        elif num < 0:
            floor_val = math.floor(num)
            ceil_val = math.ceil(num)
            if num - floor_val < ceil_val - num:
                return floor_val
            else:
                return ceil_val
        else:
            return 0

    output = closest_integer("1.5")
    return output

# Test case
print(method())  # Expected output: 2