import math
from typing import List

class CombinationCalculator:
    """
    This is a class that provides methods to calculate the number of combinations for a specific count, calculate all possible combinations, and generate combinations with a specified number of elements.
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
        :param n: The total number of elements,int.
        :param m: The number of elements in each combination,int.
        :return: The number of combinations,int.
        """
        if m > n:
            return 0
        return math.comb(n, m)

    @staticmethod
    def count_all(n: int) -> int:
        """
        Calculate the number of all possible combinations.
        :param n: The total number of elements,int.
        :return: The number of all possible combinations,int,if the number of combinations is greater than 2^63-1,return float("inf").
        """
        total_combinations = 0
        for i in range(n + 1):
            total_combinations += math.comb(n, i)
            if total_combinations > 2**63 - 1:
                return float("inf")
        return total_combinations

    def select(self, m: int) -> List[List[str]]:
        """
        Generate combinations with a specified number of elements.
        :param m: The number of elements in each combination,int.
        :return: A list of combinations,List[List[str]].
        """
        result = []
        self._select(0, [], 0, result, m)
        return result

    def select_all(self) -> List[List[str]]:
        """
        Generate all possible combinations of selecting elements from the given data list,and it uses the select method.
        :return: A list of combinations,List[List[str]].
        """
        result = []
        for m in range(1, len(self.datas) + 1):
            result.extend(self.select(m))
        return result

    def _select(self, dataIndex: int, resultList: List[str], resultIndex: int, result: List[List[str]], m: int):
        """
        Generate combinations with a specified number of elements by recursion.
        :param dataIndex: The index of the data to be selected,int.
        :param resultList: The list of elements in the combination,List[str].
        :param resultIndex: The index of the element in the combination,int.
        :param result: The list of combinations,List[List[str]].
        :param m: The number of elements in each combination,int.
        :return: None.
        """
        if resultIndex == m:
            result.append(resultList.copy())
            return
        if dataIndex == len(self.datas):
            return
        # Include the current data
        resultList[resultIndex] = self.datas[dataIndex]
        self._select(dataIndex + 1, resultList, resultIndex + 1, result, m)
        # Exclude the current data
        self._select(dataIndex + 1, resultList, resultIndex, result, m)

if __name__ == "__main__":
    # Test cases
    calc = CombinationCalculator(["A", "B", "C", "D"])

    # Test count method
    print(CombinationCalculator.count(4, 2))  # Expected output: 6

    # Test count_all method
    print(CombinationCalculator.count_all(4))  # Expected output: 15

    # Test select method
    print(calc.select(2))  # Expected output: [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]

    # Test select_all method
    print(calc.select_all())  # Expected output: [['A'], ['B'], ['C'], ['D'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D'], ['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D'], ['A', 'B', 'C', 'D']]