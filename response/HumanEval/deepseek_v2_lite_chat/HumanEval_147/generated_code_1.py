def method():
    import numpy as np

    # Define the function to calculate the sum of three elements
    def sum_of_triplets(a):
        count = 0
        n = len(a)
        # Iterate through all possible triplets
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Check if the sum of the triplet is divisible by 3
                    if (a[i] + a[j] + a[k]) % 3 == 0:
                        count += 1
        return count

    # Example usage
    n = 10
    # Generate the array 'a' based on the formula
    a = np.array([i * i - i + 1 for i in range(1, n + 1)])
    # Calculate and return the number of triples with a sum divisible by 3
    output = sum_of_triplets(a)
    return output

# Test the function with a specific example
print(method())