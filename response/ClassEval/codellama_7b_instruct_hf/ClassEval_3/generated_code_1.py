import itertools

class ArrangementCalculator:
    def __init__(self, datas):
        self.datas = datas

    @staticmethod
    def count(n, m=None):
        if m is None or m == n:
            return itertools.product(range(n + 1), repeat=m)
        else:
            return itertools.combinations(range(n + 1), m)

    @staticmethod
    def count_all(n):
        return itertools.product(range(n + 1), repeat=n)

    def select(self, m=None):
        if m is None:
            m = len(self.datas)
        return list(itertools.combinations(self.datas, m))

    def select_all(self):
        return list(itertools.product(self.datas, repeat=len(self.datas)))

    @staticmethod
    def factorial(n):
        return itertools.product(range(n + 1), repeat=n)

if __name__ == "__main__":
    ac = ArrangementCalculator([1, 2, 3])
    print(ac.count(3, 2))
    print(ac.count_all(3))
    print(ac.select(2))
    print(ac.select_all())
    print(ac.factorial(5))