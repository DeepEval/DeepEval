class BitStatusUtil:
    """
    This is a utility class that provides methods for manipulating and checking status using bitwise operations.
    """

    @staticmethod
    def add(states, stat):
        """
        Add a status to the current status, and check the parameters whether they are legal.
        :param states: Current status, int.
        :param stat: Status to be added, int.
        :return: The status after adding the status, int.
        """
        if not isinstance(states, int) or not isinstance(stat, int):
            raise ValueError("Both states and stat must be integers.")
        if states < 0 or stat < 0:
            raise ValueError("Both states and stat must be non-negative.")
        if states & stat == 0:
            raise ValueError("The specified status is not a subset of the current status.")
        return states | stat

    @staticmethod
    def has(states, stat):
        """
        Check if the current status contains the specified status, and check the parameters whether they are legal.
        :param states: Current status, int.
        :param stat: Specified status, int.
        :return: True if the current status contains the specified status, otherwise False, bool.
        """
        if not isinstance(states, int) or not isinstance(stat, int):
            raise ValueError("Both states and stat must be integers.")
        return (states & stat) != 0

    @staticmethod
    def remove(states, stat):
        """
        Remove the specified status from the current status, and check the parameters whether they are legal.
        :param states: Current status, int.
        :param stat: Specified status, int.
        :return: The status after removing the specified status, int.
        """
        if not isinstance(states, int) or not isinstance(stat, int):
            raise ValueError("Both states and stat must be integers.")
        if states < 0 or stat < 0:
            raise ValueError("Both states and stat must be non-negative.")
        mask = ~stat
        return states & mask

    @staticmethod
    def check(args):
        """
        Check if the parameters are legal, args must be greater than or equal to 0 and must be even, if not, raise ValueError.
        :param args: Parameters to be checked, list.
        :return: None.
        """
        if not all(isinstance(arg, int) and arg >= 0 and arg % 2 == 0 for arg in args):
            raise ValueError("All arguments must be non-negative even integers.")


# Example usage of the methods
if __name__ == "__main__":
    bit_status_util = BitStatusUtil()
    print(bit_status_util.add(6, 2))  # Expected output: 6
    print(bit_status_util.has(6, 2))  # Expected output: True
    print(bit_status_util.remove(6, 2))  # Expected output: 4
    try:
        bit_status_util.check([2, 3, 4])  # Expected output: None
    except ValueError as e:
        print(e)  # Expected output: 3 not even