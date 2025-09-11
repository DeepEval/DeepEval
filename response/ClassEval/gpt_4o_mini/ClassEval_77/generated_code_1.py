import random

class Snake:
    """
    The class is a snake game, which allows the snake to move and eat food, and also enables to reset and generate a random food position.
    """

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        """
        Initialize the length of the snake, screen width, screen height, block size, snake head position, score, and food position.
        :param SCREEN_WIDTH: int
        :param SCREEN_HEIGHT: int
        :param BLOCK_SIZE: int, Size of moving units
        :param food_position: tuple, representing the position(x, y) of food.
        """
        self.length = 1
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BLOCK_SIZE = BLOCK_SIZE
        self.positions = [(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)]
        self.score = 0
        self.food_position = food_position

    def move(self, direction):
        """
        Move the snake in the specified direction.
        :param direction: tuple, representing the direction of movement (x, y).
        :return: None
        """
        # Calculate new head position
        new_head = (self.positions[0][0] + direction[0] * self.BLOCK_SIZE,
                     self.positions[0][1] + direction[1] * self.BLOCK_SIZE)

        # Check for food consumption
        if new_head == self.food_position:
            self.eat_food()

        # Check for collision with self
        if new_head in self.positions:
            self.reset()
        else:
            self.positions.insert(0, new_head)  # Add new head position
            if len(self.positions) > self.length:
                self.positions.pop()  # Remove the tail

    def random_food_position(self):
        """
        Randomly generate a new food position, but don't place it on the snake.
        :return: None, Change the food position
        """
        while True:
            x = random.randint(0, (self.SCREEN_WIDTH // self.BLOCK_SIZE) - 1) * self.BLOCK_SIZE
            y = random.randint(0, (self.SCREEN_HEIGHT // self.BLOCK_SIZE) - 1) * self.BLOCK_SIZE
            self.food_position = (x, y)
            if self.food_position not in self.positions:
                break

    def reset(self):
        """
        Reset the snake to its initial state.
        :return: None
        """
        self.length = 1
        self.positions = [(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)]
        self.score = 0
        self.random_food_position()

    def eat_food(self):
        """
        Increase the length of the snake by 1 and increase the score by 100.
        :return: None
        """
        self.length += 1
        self.score += 100
        self.random_food_position()

# Test cases for each method
if __name__ == "__main__":
    # Initialize the snake game
    snake = Snake(100, 100, 10, (50, 50))

    # Test random_food_position
    snake.random_food_position()
    print("Random food position:", snake.food_position)

    # Test move
    snake.move((1, 0))  # Move right
    print("Snake length after move:", snake.length)
    print("Snake positions after move:", snake.positions)

    # Test eat_food
    snake.food_position = (60, 50)  # Set food position to be eaten
    snake.move((1, 0))  # Move right again to eat food
    print("Snake length after eating food:", snake.length)
    print("Snake score after eating food:", snake.score)
    print("Snake positions after eating food:", snake.positions)

    # Test reset
    snake.reset()
    print("Snake length after reset:", snake.length)
    print("Snake positions after reset:", snake.positions)
    print("Snake score after reset:", snake.score)
    print("New food position after reset:", snake.food_position)