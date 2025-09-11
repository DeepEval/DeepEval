class BitStatusUtil:
    """
    This is a utility class that provides methods for manipulating and checking
    status using bitwise operations.
    """

    @staticmethod
    def add(states, stat):
        """
        Add a status to the current status,and check the parameters wheather they
        are legal.
        :param states: Current status,int.
        :param stat: Status to be added,int.
        :return: The status after adding the status,int.
        """
        BitStatusUtil.check([states, stat])
        return states | stat

    @staticmethod
    def has(states, stat):
        """
        Check if the current status contains the specified status,and check the
        parameters wheather they are legal.
        :param states: Current status,int.
        :param stat: Specified status,int.
        :return: True if the current status contains the specified status,otherwise
        False,bool.
        """
        BitStatusUtil.check([states, stat])
        return bool(states & stat)

    @staticmethod
    def remove(states, stat):
        """
        Remove the specified status from the current status,and check the parameters
        wheather they are legal.
        :param states: Current status,int.
        :param stat: Specified status,int.
        :return: The status after removing the specified status,int.
        """
        BitStatusUtil.check([states, stat])
        return states & ~stat

    @staticmethod
    def check(args):
        """
        Check if the parameters are legal, args must be greater than or equal to
        0 and must be even,if not,raise ValueError.
        :param args: Parameters to be checked,list.
        :return: None.
        """
        for arg in args:
            if not isinstance(arg, int) or arg < 0 or arg % 2!= 0:
                raise ValueError(f"{arg} not even")

if __name__ == "__main__":
    bit_status_util = BitStatusUtil()
    output = bit_status_util.add(2, 4)
    print("Test case for add:", output)
    
    output = bit_status_util.has(6, 2)
    print("Test case for has:", output)
    
    output = bit_status_util.remove(6, 2)
    print("Test case for remove:", output)
    
    bit_status_util.check([2, 4, 6])