import random

class Snake:
    """
    The class is a snake game, which allows the snake to move and eat food, and also enables resetting, and generating a random food position.
    """

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        self.length = 1
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BLOCK_SIZE = BLOCK_SIZE
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.score = 0
        self.food_position = food_position

    def move(self, direction):
        # Calculate new head position
        new_head = (self.positions[0][0] + direction[0] * self.BLOCK_SIZE,
                    self.positions[0][1] + direction[1] * self.BLOCK_SIZE)

        # Check if the new head is on the food position
        if new_head == self.food_position:
            self.eat_food()
        else:
            # Move the snake
            self.positions = [new_head] + self.positions[:-1]

        # Check for collision with itself
        if len(self.positions) != len(set(self.positions)):
            self.reset()

    def random_food_position(self):
        while True:
            x = random.randint(0, (self.SCREEN_WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            y = random.randint(0, (self.SCREEN_HEIGHT - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            food_position = (x, y)
            if food_position not in self.positions:
                self.food_position = food_position
                break

    def reset(self):
        self.length = 1
        self.positions = [((self.SCREEN_WIDTH / 2), (self.SCREEN_HEIGHT / 2))]
        self.score = 0
        self.random_food_position()

    def eat_food(self):
        self.length += 1
        self.score += 100
        self.positions.append(self.positions[-1])  # Extend the snake
        self.random_food_position()

if __name__ == "__main__":
    # Test case for move
    snake = Snake(100, 100, 10, (60, 50))
    snake.move((1, 0))
    print(f"After move: {snake.positions}, score: {snake.score}")

    # Test case for random_food_position
    snake.random_food_position()
    print(f"New food position: {snake.food_position}")

    # Test case for reset
    snake.reset()
    print(f"After reset: {snake.positions}, score: {snake.score}")

    # Test case for eat_food
    snake.eat_food()
    print(f"After eat_food: length: {snake.length}, score: {snake.score}, food_position: {snake.food_position}")