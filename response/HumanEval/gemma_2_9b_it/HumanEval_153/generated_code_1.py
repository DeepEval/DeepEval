def method(className, extensions):
    strongest_extension = None
    max_strength = float('-inf')  

    for extension in extensions:
        cap_count = sum(1 for char in extension if char.isupper())
        sm_count = sum(1 for char in extension if char.islower())
        strength = cap_count - sm_count

        if strength > max_strength:
            max_strength = strength
            strongest_extension = extension

    output = f"{className}.{strongest_extension}"
    return output

# Test Case
className = "Slices"
extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
result = method(className, extensions)
print(result)  # Output: Slices.SErviNGSliCes