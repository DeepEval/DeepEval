def intersect_interval(interval1, interval2):
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    
    x_intersect_min = max(x1_min, x2_min)
    x_intersect_max = min(x1_max, x2_max)
    
    if x_intersect_min <= x_intersect_max:
        return (x_intersect_min, x_intersect_max)
    else:
        return (0, 0)

if __name__ == "__main__":
    interval1 = (1, 5)
    interval2 = (3, 7)
    result = intersect_interval(interval1, interval2)
    print(result)  # Expected output: (3, 5)

    interval3 = (6, 8)
    interval4 = (1, 2)
    result = intersect_interval(interval3, interval4)
    print(result)  # Expected output: (0, 0)