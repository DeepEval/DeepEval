import numpy as np

def method():
    # Create two arrays x and y
    x = np.array([12, 24, 36, 48, 60])
    y = np.array([4, 6, 8, 10, 12])

    # Perform element-wise division
    quotient = x / y

    # Calculate the remainder using the modulus operator (%)
    remainder = x % y

    # Return the output (optional)
    return quotient, remainder

# Call the method for validation
quotient, remainder = method()

print("Element-wise Division (Quotient):", quotient)
print("Element-wise Division (Remainder):", remainder)