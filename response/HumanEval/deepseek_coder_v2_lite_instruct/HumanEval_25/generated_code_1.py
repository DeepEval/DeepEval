def method():
    def prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def count_factors(factors):
        count_dict = {}
        for factor in factors:
            if factor in count_dict:
                count_dict[factor] += 1
            else:
                count_dict[factor] = 1
        return count_dict

    def format_output(count_dict):
        output = []
        for factor, count in count_dict.items():
            output.extend([factor] * count)
        return output

    # Example input
    n = 360

    # Get prime factors
    factors = prime_factors(n)

    # Count the occurrences of each factor
    count_dict = count_factors(factors)

    # Format the output
    output = format_output(count_dict)

    return output

# Test case
print(method())  # Expected output: [2, 2, 2, 3, 3, 5]