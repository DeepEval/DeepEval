import itertools

class ArrangementCalculator:
    """
    The Arrangement class provides permutation calculations and selection operations for a given set of data elements.
    """

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
            return int(itertools.factorial(n))
        else:
            return int(itertools.permute([i for i in range(n)]))

    @staticmethod
    def count_all(n):
        """
        Counts the total number of all possible arrangements by choosing at least 1 item and at most n items from n items.
        :param n: int, the total number of items.
        :return: int, the count of all arrangements.
        """
        return sum(map(int, [ArrangementCalculator.count(n, m) for m in range(n + 1)]))

    def select(self, m=None):
        """
        Generates a list of arrangements by selecting m items from the internal datas.
        If m is not provided, selects all items.
        :param m: int, the number of items to be chosen (default=None).
        :return: List, a list of arrangements.
        """
        if m is None:
            m = len(self.datas)
        return [list(itertools.permutations(self.datas, m)) for _ in range(n)]

    def select_all(self):
        """
        Generates a list of all arrangements by selecting at least 1 item and at most the number of internal datas.
        :return: List, a list of all arrangements.
        """
        return self.select()

    @staticmethod
    def factorial(n):
        """
        Calculates the factorial of a given number.
        :param n: int, the number to calculate the factorial.
        :return: int, the factorial of the given number.
        """
        return int(itertools.factorial(n))

# Test cases
if __name__ == "__main__":
    # Test Case for count
    print(ArrangementCalculator.count(5, 3))  # Should print 60
    print(ArrangementCalculator.count(5))      # Should print 120

    # Test Case for count_all
    print(ArrangementCalculator.count_all(4))  # Should print 24

    # Test Case for select
    ac = ArrangementCalculator([1, 2, 3, 4])
    print(ac.select(2))     # Should print [[1, 2], [1, 3], [1, 4], [2, 1], [2, 3], [2, 4], [3, 1], [3, 2], [3, 4], [4, 1], [4, 2], [4, 3]]

    # Test Case for select_all
    print(ac.select_all())  # Should print [[1], [2], [3], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

    # Test Case for factorial
    print(ArrangementCalculator.factorial(4))  # Should print 24