from typing import Tuple

def intersect_intervals(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> Tuple[float, float]:
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    
    if x1_min > x2_max or x2_min > x1_max:
        return (0, 0)  # No intersection
    
    return (max(x1_min, x2_min), min(x1_max, x2_max))

if __name__ == "__main__":
    # Example usage:
    interval1 = (1, 5)
    interval2 = (3, 8)
    
    intersection = intersect_intervals(interval1, interval2)
    
    print(f"Intersection of {interval1} and {interval2}: {intersection}")