def intersect_interavl(interval1, interval2):
    x_intersect = (max(interval1[0], interval2[0]), min(interval1[1], interval2[1]))
    return x_intersect if x_intersect[0] <= x_intersect[1] else (0, 0)

if __name__ == "__main__":
    # Sample input values
    interval1 = (5, 10)
    interval2 = (7, 12)

    # Call the function and print the results
    result = intersect_interavl(interval1, interval2)
    print(result)