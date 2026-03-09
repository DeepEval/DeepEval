def intersect_interval(interval1: tuple[float, float], interval2: tuple[float, float]) -> tuple[float, float]:
  x1_min, x1_max = interval1
  x2_min, x2_max = interval2
  if x1_max < x2_min or x2_max < x1_min:
    return (0, 0)
  else:
    return (max(x1_min, x2_min), min(x1_max, x2_max))

if __name__ == "__main__":
  interval1 = (1, 5)
  interval2 = (3, 7)
  x_intersect = intersect_interval(interval1, interval2)
  print(f"Intersection: {x_intersect}")

  interval1 = (1, 3)
  interval2 = (5, 7)
  x_intersect = intersect_interval(interval1, interval2)
  print(f"Intersection: {x_intersect}")