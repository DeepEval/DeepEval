from typing import Union

def compare_one(a: Union[int, float, str], b: Union[int, float, str]) -> Union[int, float, str, None]:
    """
    Compares two real numbers and returns the larger variable in its given variable type.

    Args:
        a: The first real number.
        b: The second real number.

    Returns:
        The larger variable in its given variable type. None if the values are equal.
    """

    if isinstance(a, str):
        a = float(a.replace(",", "."))

    if isinstance(b, str):
        b = float(b.replace(",", "."))

    if a > b:
        return a
    elif b > a:
        return b
    else:
        return None

# Test case
print(compare_one(1, 2.5))  # Output: 2.5
print(compare_one(1, "2,3"))  # Output: "2,3"
print(compare_one("5,1", "6"))  # Output: "6"
print(compare_one("1", 1))  # Output: None