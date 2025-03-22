import math

def method():
    def calculate_collisions(n):
        # Initialize a set to store the distances traveled by the cars
        left_to_right = set()
        right_to_left = set()

        collisions = 0
        for i in range(1, n + 1):
            # Calculate the distance traveled by the car moving left to right
            left_to_right.add(i)
            # Calculate the distance traveled by the car moving right to left
            right_to_left.add(n - i + 1)

            # Check for collisions
            for j in range(i):
                if n - (j + 1) + 1 in left_to_right:
                    collisions += 1

        return collisions

    n = 5  # number of cars
    return calculate_collisions(n)


output = method()
print(output)