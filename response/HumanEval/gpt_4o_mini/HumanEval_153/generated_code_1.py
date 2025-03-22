def method(class_name, extensions):
    def calculate_strength(extension):
        CAP = sum(1 for c in extension if c.isupper())
        SM = sum(1 for c in extension if c.islower())
        return CAP - SM

    strongest_extension = None
    strongest_strength = float('-inf')  # Start with the lowest possible strength

    for extension in extensions:
        strength = calculate_strength(extension)
        if strength > strongest_strength:
            strongest_strength = strength
            strongest_extension = extension

    return f"{class_name}.{strongest_extension}"

# Test case for validation
if __name__ == "__main__":
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
    output = method(class_name, extensions)
    print(output)  # Should print: Slices.SErviNGSliCes