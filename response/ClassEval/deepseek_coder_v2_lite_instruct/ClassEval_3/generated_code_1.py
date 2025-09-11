import itertools

class ArrangementCalculator:
    """
    The Arrangement class provides permutation calculations and selection operations for a given set of data elements.
    """

    def __init__(self, datas):
        """
        Initializes the ArrangementCalculator object with a list of datas.
        :param datas: List, the data elements to be used for arrangements.
        """
        self.datas = datas

    @staticmethod
    def count(n, m=None):
        """
        Counts the number of arrangements by choosing m items from n items (permutations).
        If m is not provided or n equals m, returns factorial(n).
        :param n: int, the total number of items.
        :param m: int, the number of items to be chosen (default=None).
        :return: int, the count of arrangements.
        """
        if m is None or n == m:
            return ArrangementCalculator.factorial(n)
        else:
            return ArrangementCalculator.factorial(n) // ArrangementCalculator.factorial(n - m)

    @staticmethod
    def count_all(n):
        """
        Counts the total number of all possible arrangements by choosing at least 1 item and at most n items from n items.
        :param n: int, the total number of items.
        :return: int, the count of all arrangements.
        """
        total_count = 0
        for i in range(1, n + 1):
            total_count += ArrangementCalculator.count(n, i)
        return total_count

    def select(self, m=None):
        """
        Generates a list of arrangements by selecting m items from the internal datas.
        If m is not provided, selects all items.
        :param m: int, the number of items to be chosen (default=None).
        :return: List, a list of arrangements.
        """
        if m is None:
            m = len(self.datas)
        return [list(i) for i in itertools.permutations(self.datas, m)]

    def select_all(self):
        """
        Generates a list of all arrangements by selecting at least 1 item and at most the number of internal datas.
        :return: List, a list of all arrangements.
        """
        return [list(i) for i in itertools.permutations(self.datas)]

    @staticmethod
    def factorial(n):
        """
        Calculates the factorial of a given number.
        :param n: int, the number to calculate the factorial.
        :return: int, the factorial of the given number.
        """
        if n == 0:
            return 1
        else:
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

# Test cases
if __name__ == "__main__":
    # Test count method
    print(ArrangementCalculator.count(5, 3))  # Expected output: 60
    print(ArrangementCalculator.count(5))      # Expected output: 120

    # Test count_all method
    print(ArrangementCalculator.count_all(4))  # Expected output: 64

    # Test select method
    ac = ArrangementCalculator([1, 2, 3, 4])
    print(ac.select(2))  # Expected output: [[1, 2], [1, 3], [1, 4], [2, 1], [2, 3], [2, 4], [3, 1], [3, 2], [3, 4], [4, 1], [4, 2], [4, 3]]

    # Test select_all method
    ac = ArrangementCalculator([1, 2, 3])
    print(ac.select_all())  # Expected output: [[1], [2], [3], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

    # Test factorial method
    print(ArrangementCalculator.factorial(4))  # Expected output: 24