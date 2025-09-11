from collections import Counter

class DataStatistics:
    def mean(self, data):
        count = Counter(data)
        mean = sum(count.values()) / len(data)
        return mean

    def median(self, data):
        count = Counter(data)
        median = sorted(count.values(), key=lambda x: x[1])[len(data) // 2]
        return median

    def mode(self, data):
        count = Counter(data)
        mode = [value for key, value in count.most_common(1)]
        return mode

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    ds = DataStatistics()
    print("Mean:", ds.mean(data))
    print("Median:", ds.median(data))
    print("Mode:", ds.mode(data))