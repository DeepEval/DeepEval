def method(class_name, extensions):
    """
    This function finds the strongest extension for a given class and returns the class name concatenated with the strongest extension.

    Args:
    class_name (str): The name of the class.
    extensions (list): A list of extensions.

    Returns:
    str: The class name concatenated with the strongest extension.
    """

    # Initialize the strongest extension and its strength
    strongest_extension = None
    strongest_strength = float('-inf')  # Initialize with negative infinity

    # Iterate over each extension
    for extension in extensions:
        # Calculate the strength of the current extension
        cap = sum(1 for char in extension if char.isupper())
        sm = sum(1 for char in extension if char.islower())
        strength = cap - sm

        # If the strength of the current extension is greater than the strongest strength found so far,
        # update the strongest extension and its strength
        if strength > strongest_strength:
            strongest_extension = extension
            strongest_strength = strength

    # Return the class name concatenated with the strongest extension
    return f"{class_name}.{strongest_extension}"


# Test case
class_name = "Slices"
extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
print(method(class_name, extensions))  # Output: Slices.SErviNGSliCes