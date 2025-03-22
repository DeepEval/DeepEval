def method():
    numbers = [12, 45, 23, 98, 7, 32, 17, 53, 11]  
    largest_prime = -1 

    for num in numbers:
      # Check if the number is prime
      if num > 1:
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
          if (num % i) == 0:
            is_prime = False
            break
        if is_prime and num > largest_prime:
          largest_prime = num

    # Calculate the sum of digits of the largest prime
    sum_of_digits = 0
    while largest_prime > 0:
      digit = largest_prime % 10
      sum_of_digits += digit
      largest_prime //= 10

    return sum_of_digits

output = method()
print(f"Sum of digits of the largest prime: {output}")