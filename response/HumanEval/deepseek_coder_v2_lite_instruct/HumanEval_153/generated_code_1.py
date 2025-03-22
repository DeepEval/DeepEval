def method():
    # Given inputs
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']

    # Initialize variables to store the strongest extension
    strongest_extension = None
    max_strength = float('-inf')

    # Iterate over each extension to calculate its strength
    for extension in extensions:
        # Count CAP and SM
        CAP = sum(1 for c in extension if c.isupper())
        SM = sum(1 for c in extension if c.islower())
        strength = CAP - SM

        # Update the strongest extension if this one is stronger
        if strength > max_strength:
            max_strength = strength
            strongest_extension = extension

    # Return the result in the required format
    output = f"{class_name}.{strongest_extension}"
    return output

# Test case
print(method())  # Expected output: 'Slices.SErviNGSliCes'