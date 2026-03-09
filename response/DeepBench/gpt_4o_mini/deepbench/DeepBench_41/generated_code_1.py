from typing import Tuple

def intersect_interval(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> Tuple[float, float]:
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    x_intersect = (max(x1_min, x2_min), min(x1_max, x2_max))
    if x_intersect[0] > x_intersect[1]:
        return (0, 0)
    return x_intersect

if __name__ == "__main__":
    interval1 = (1, 5)
    interval2 = (3, 7)
    result = intersect_interval(interval1, interval2)
    print("Intersection:", result)  # Expected output: Intersection: (3, 5)

    interval1 = (1, 2)
    interval2 = (3, 4)
    result = intersect_interval(interval1, interval2)
    print("Intersection:", result)  # Expected output: Intersection: (0, 0)