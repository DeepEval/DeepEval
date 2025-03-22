def method(class_name, extensions):
    def extension_strength(extension):
        # Calculate the number of uppercase and lowercase letters
        CAP = sum(1 for c in extension if c.isupper())
        SM = sum(1 for c in extension if c.islower())
        # Calculate the strength
        return CAP - SM

    # Initialize variables to keep track of the strongest extension
    strongest_extension = None
    max_strength = float('-inf')

    # Iterate over the extensions to find the strongest one
    for ext in extensions:
        strength = extension_strength(ext)
        if strength > max_strength:
            max_strength = strength
            strongest_extension = ext

    # Format the result as specified
    output = f"{class_name}.{strongest_extension}"
    return output

# Test case
class_name = "Slices"
extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
output = method(class_name, extensions)
print(output)  # Expected: 'Slices.SErviNGSliCes'