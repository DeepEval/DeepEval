# Import necessary packages
import numpy as np

def method():
    # Input number of cars
    # n = int(input())

    # 修改：
    n = 4

    # Initialize lists to store positions and velocities of cars
    left_cars = []
    right_cars = []

    # Define a collision detection function
    def check_collision():
        for left_car in left_cars:
            for right_car in right_cars:
                if abs(left_car[0] - right_car[0]) <= 1 and left_car[1] * right_car[1] > 0:
                    return True
        return False

    # Main simulation loop
    collisions = 0
    while True:
        # Move cars
        for car in left_cars:
            car[0] -= 1
        for car in right_cars:
            car[0] += 1

        # Check for collisions
        if check_collision():
            collisions += 1

        # Check if cars have reached the end of the road
        if max(left_cars) == 0 or min(right_cars) == n:
            break

    # Return the number of collisions
    return collisions

# Test case
test_case = [[0, -1], [1, 1], [2, -1], [3, 1]]
print(method())