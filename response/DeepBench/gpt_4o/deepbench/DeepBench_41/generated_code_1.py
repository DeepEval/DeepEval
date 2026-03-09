from typing import Tuple

def intersect_interval(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> Tuple[float, float]:
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    
    # Calculate the intersection
    intersect_min = max(x1_min, x2_min)
    intersect_max = min(x1_max, x2_max)
    
    # Check if the intersection is valid
    if intersect_min >= intersect_max:
        return (0, 0)  # Represents the empty set
    
    return (intersect_min, intersect_max)

if __name__ == "__main__":
    # Example usage
    interval1 = (1.0, 5.0)
    interval2 = (3.0, 6.0)
    result = intersect_interval(interval1, interval2)
    print(f"Intersection of {interval1} and {interval2} is: {result}")
    
    # Another test case where the intervals do not intersect
    interval3 = (1.0, 2.0)
    interval4 = (3.0, 4.0)
    result2 = intersect_interval(interval3, interval4)
    print(f"Intersection of {interval3} and {interval4} is: {result2}")