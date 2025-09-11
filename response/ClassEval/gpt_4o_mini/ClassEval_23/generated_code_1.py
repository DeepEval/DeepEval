import math
from typing import List
from itertools import combinations

class CombinationCalculator:
    """
    This is a class that provides methods to calculate the number of combinations for a specific count,
    calculate all possible combinations, and generate combinations with a specified number of elements.
    """

    def __init__(self, datas: List[str]):
        """
        Initialize the calculator with a list of data.
        """
        self.datas = datas

    @staticmethod
    def count(n: int, m: int) -> int:
        """
        Calculate the number of combinations for a specific count.
        :param n: The total number of elements, int.
        :param m: The number of elements in each combination, int.
        :return: The number of combinations, int.
        >>> CombinationCalculator.count(4, 2)
        6
        """
        if m > n or m < 0:
            return 0
        return math.comb(n, m)

    @staticmethod
    def count_all(n: int) -> int:
        """
        Calculate the number of all possible combinations.
        :param n: The total number of elements, int.
        :return: The number of all possible combinations, int,
                if the number of combinations is greater than 2^63-1, return float("inf").
        >>> CombinationCalculator.count_all(4)
        15
        """
        total_combinations = (1 << n) - 1  # 2^n - 1
        return total_combinations if total_combinations <= (2**63 - 1) else float("inf")

    def select(self, m: int) -> List[List[str]]:
        """
        Generate combinations with a specified number of elements.
        :param m: The number of elements in each combination, int.
        :return: A list of combinations, List[List[str]].
        >>> calc = CombinationCalculator(["A", "B", "C", "D"])
        >>> calc.select(2)
        [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]
        """
        return [list(combo) for combo in combinations(self.datas, m)]

    def select_all(self) -> List[List[str]]:
        """
        Generate all possible combinations of selecting elements from the given data list, and it uses the select method.
        :return: A list of combinations, List[List[str]].
        >>> calc = CombinationCalculator(["A", "B", "C", "D"])
        >>> calc.select_all()
        [['A'], ['B'], ['C'], ['D'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D'], ['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D'], ['A', 'B', 'C', 'D']]
        """
        all_combinations = []
        for m in range(1, len(self.datas) + 1):
            all_combinations.extend(self.select(m))
        return all_combinations

    def _select(self, dataIndex: int, resultList: List[str], resultIndex: int, result: List[List[str]]):
        """
        Generate combinations with a specified number of elements by recursion.
        :param dataIndex: The index of the data to be selected, int.
        :param resultList: The list of elements in the combination, List[str].
        :param resultIndex: The index of the element in the combination, int.
        :param result: The list of combinations, List[List[str]].
        :return: None.
        >>> calc = CombinationCalculator(["A", "B", "C", "D"])
        >>> result = []
        >>> calc._select(0, [None] * 2, 0, result)
        >>> result
        [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]
        """
        if resultIndex == len(resultList):
            result.append(list(resultList))
            return
        
        for i in range(dataIndex, len(self.datas)):
            resultList[resultIndex] = self.datas[i]
            self._select(i + 1, resultList, resultIndex + 1, result)

if __name__ == "__main__":
    # Test cases
    # Test count method
    print("Count of combinations (4 choose 2):", CombinationCalculator.count(4, 2))  # Expected: 6

    # Test count_all method
    print("Count of all combinations (4):", CombinationCalculator.count_all(4))  # Expected: 15

    # Test select method
    calc = CombinationCalculator(["A", "B", "C", "D"])
    print("Select combinations (2):", calc.select(2))  # Expected combinations of 2

    # Test select_all method
    print("Select all combinations:", calc.select_all())  # Expected all combinations

    # Test _select method
    result = []
    calc._select(0, [None] * 2, 0, result)
    print("Recursive select combinations (2):", result)  # Expected combinations of 2