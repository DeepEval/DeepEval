import math

class BitStatusUtil:
    """
    A utility class for manipulating and checking status using bitwise operations.
    """

    @staticmethod
    def add(states, stat):
        """
        Add a status to the current status, and check the parameters whether they are legal.
        :param states: Current status, int.
        :param stat: Status to be added, int.
        :return: The status after adding the status, int.
        """
        if states < 0 or stat < 0:
            raise ValueError("Negative values are not allowed")
        if math.ceil(math.log(states, 2)) != math.ceil(math.log(stat, 2)):
            raise ValueError("The status to be added must be a power of 2")
        return states + stat

    @staticmethod
    def has(states, stat):
        """
        Check if the current status contains the specified status, and check the parameters whether they are legal.
        :param states: Current status, int.
        :param stat: Specified status, int.
        :return: True if the current status contains the specified status, otherwise False, bool.
        """
        if states < 0 or stat < 0:
            raise ValueError("Negative values are not allowed")
        if math.ceil(math.log(states, 2)) != math.ceil(math.log(stat, 2)):
            raise ValueError("The status to be checked must be a power of 2")
        return states & stat == stat

    @staticmethod
    def remove(states, stat):
        """
        Remove the specified status from the current status, and check the parameters whether they are legal.
        :param states: Current status, int.
        :param stat: Specified status, int.
        :return: The status after removing the specified status, int.
        """
        if states < 0 or stat < 0:
            raise ValueError("Negative values are not allowed")
        if math.ceil(math.log(states, 2)) != math.ceil(math.log(stat, 2)):
            raise ValueError("The status to be removed must be a power of 2")
        return states & ~stat

    @staticmethod
    def check(args):
        """
        Check if the parameters are legal, args must be greater than or equal to 0 and must be even, if not, raise ValueError.
        :param args: Parameters to be checked, list.
        :return: None.
        """
        for arg in args:
            if arg < 0 or math.ceil(math.log(arg, 2)) != math.ceil(math.log(arg, 2)):
                raise ValueError("The status must be a power of 2")
            
if __name__ == "__main__":
    # Example usage
    util = BitStatusUtil()
    current_status = 0b0001  # 1 in binary
    status_to_add = 0b0010   # 2 in binary

    new_status = util.add(current_status, status_to_add)
    print(f"New Status after adding: {bin(new_status)}")  # Should print 0b0011

    has_status = util.has(new_status, status_to_add)
    print(f"Has status {bin(status_to_add)}: {has_status}")  # Should print True

    removed_status = util.remove(new_status, status_to_add)
    print(f"Status after removing {bin(status_to_add)}: {bin(removed_status)}")  # Should print 0b0001

    util.check([0b0001, 0b0010])  # Should not raise an error
    