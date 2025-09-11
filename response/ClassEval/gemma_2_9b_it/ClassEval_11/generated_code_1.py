class BitStatusUtil:
    """
    This is a utility class that provides methods for manipulating and checking status using bitwise operations.
    """

    @staticmethod
    def add(states, stat):
        """
        Add a status to the current status,and check the parameters wheather they are legal.
        :param states: Current status,int.
        :param stat: Status to be added,int.
        :return: The status after adding the status,int.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.add(2,4)
        6

        """
        BitStatusUtil.check([states, stat])
        return states | stat

    @staticmethod
    def has(states, stat):
        """
        Check if the current status contains the specified status,and check the parameters wheather they are legal.
        :param states: Current status,int.
        :param stat: Specified status,int.
        :return: True if the current status contains the specified status,otherwise False,bool.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.has(6,2)
        True

        """
        BitStatusUtil.check([states, stat])
        return (states & stat) != 0

    @staticmethod
    def remove(states, stat):
        """
        Remove the specified status from the current status,and check the parameters wheather they are legal.
        :param states: Current status,int.
        :param stat: Specified status,int.
        :return: The status after removing the specified status,int.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.remove(6,2)
        4

        """
        BitStatusUtil.check([states, stat])
        return states & ~stat

    @staticmethod
    def check(args):
        """
        Check if the parameters are legal, args must be greater than or equal to 0 and must be even,if not,raise ValueError.
        :param args: Parameters to be checked,list.
        :return: None.
        >>> bit_status_util = BitStatusUtil()
        >>> bit_status_util.check([2,3,4])
        Traceback (most recent call last):
        ...
        ValueError: 3 not even
        """
        for arg in args:
            if arg < 0 or arg % 2 != 0:
                raise ValueError(f"{arg} not even")



if __name__ == "__main__":
    instance = BitStatusUtil()
    # Test case for add
    output = instance.add(2, 4)
    print(f"add(2, 4) = {output}")  # Expected output: 6

    # Test case for has
    output = instance.has(6, 2)
    print(f"has(6, 2) = {output}")  # Expected output: True

    # Test case for remove
    output = instance.remove(6, 2)
    print(f"remove(6, 2) = {output}")  # Expected output: 4

    # Test case for check
    try:
        instance.check([2, 3, 4])
    except ValueError as e:
        print(e)  # Expected output: 3 not even