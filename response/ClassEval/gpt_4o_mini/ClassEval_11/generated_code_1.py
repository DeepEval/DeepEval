class BitStatusUtil:
    """
    This is a utility class that provides methods for manipulating and checking status using bitwise operations.
    """

    @staticmethod
    def add(states, stat):
        """
        Add a status to the current status, and check if the parameters are legal.
        :param states: Current status, int.
        :param stat: Status to be added, int.
        :return: The status after adding the status, int.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.add(2, 4)
        6
        """
        BitStatusUtil.check([states, stat])  # Validate inputs
        return states | stat

    @staticmethod
    def has(states, stat):
        """
        Check if the current status contains the specified status, and check if the parameters are legal.
        :param states: Current status, int.
        :param stat: Specified status, int.
        :return: True if the current status contains the specified status, otherwise False, bool.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.has(6, 2)
        True
        """
        BitStatusUtil.check([states, stat])  # Validate inputs
        return (states & stat) == stat

    @staticmethod
    def remove(states, stat):
        """
        Remove the specified status from the current status, and check if the parameters are legal.
        :param states: Current status, int.
        :param stat: Specified status, int.
        :return: The status after removing the specified status, int.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.remove(6, 2)
        4
        """
        BitStatusUtil.check([states, stat])  # Validate inputs
        return states & ~stat

    @staticmethod
    def check(args):
        """
        Check if the parameters are legal, args must be greater than or equal to 0 and must be even,
        if not, raise ValueError.
        :param args: Parameters to be checked, list.
        :return: None.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.check([2, 3, 4])
        Traceback (most recent call last):
        ...
        ValueError: 3 not even
        """
        for arg in args:
            if arg < 0:
                raise ValueError(f"{arg} is negative")
            if arg % 2 != 0:
                raise ValueError(f"{arg} not even")

# Test cases for each method
if __name__ == "__main__":
    bit_status_util = BitStatusUtil()

    # Test add method
    output_add = bit_status_util.add(2, 4)
    print(f"Add Output: {output_add}")  # Expected output: 6

    # Test has method
    output_has = bit_status_util.has(6, 2)
    print(f"Has Output: {output_has}")  # Expected output: True

    # Test remove method
    output_remove = bit_status_util.remove(6, 2)
    print(f"Remove Output: {output_remove}")  # Expected output: 4

    # Test check method
    try:
        bit_status_util.check([2, 3, 4])
    except ValueError as ve:
        print(f"Check Output: {ve}")  # Expected output: "3 not even"

    # Check for valid inputs
    try:
        bit_status_util.check([0, 2, 4])  # Should not raise an error
        print("Check Output: All inputs are valid.")  # Expected output
    except ValueError as ve:
        print(f"Check Output: {ve}")