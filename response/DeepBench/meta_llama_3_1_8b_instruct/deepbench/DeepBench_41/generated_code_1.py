def intersect_interval(interval1, interval2):
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    x_intersect = (max(x1_min, x2_min), min(x1_max, x2_max))
    return x_intersect if x1_max >= x2_min and x2_max >= x1_min else (0, 0)

if __name__ == "__main__":
    interval1 = (1.0, 5.0)
    interval2 = (2.0, 6.0)
    print(intersect_interval(interval1, interval2))  # Expected output: (2.0, 5.0)

    interval3 = (3.0, 7.0)
    interval4 = (9.0, 10.0)
    print(intersect_interval(interval3, interval4))  # Expected output: (0, 0)