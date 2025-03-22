def method():
    interval1 = (1, 5)  
    interval2 = (3, 7)
    
    # Calculate intersection
    start_intersection = max(interval1[0], interval2[0])
    end_intersection = min(interval1[1], interval2[1])
    intersection_length = end_intersection - start_intersection + 1

    # Check if intersection length is prime
    if intersection_length > 1:
      is_prime = True
      for i in range(2, int(intersection_length**0.5) + 1):
        if intersection_length % i == 0:
          is_prime = False
          break
    else:
      is_prime = False

    output = "The length of intersection is prime" if is_prime else "The length of intersection is not prime"
    
    return output

# Test Case
print(method())  # Output: The length of intersection is prime