import random

class Snake:
    """
    The class is a snake game, with allows snake to move and eat food, and also enables to reset, and generat a random food position.
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
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.score = 0
        self.food_position = food_position


    def move(self, direction):
        """
        Move the snake in the specified direction. If the new position of the snake's head is equal to the position of the food, then eat the food; If the position of the snake's head is equal to the position of its body, then start over, otherwise its own length plus one.
        :param direction: tuple, representing the direction of movement (x, y).
        :return: None
        >>> snake.move((1,1))
        self.length = 1
        self.positions = [(51, 51), (50, 50)]
        self.score = 10
        """
        # Get the current head position
        head = self.positions[-1]
        
        # Calculate the new head position
        new_head = (head[0] + direction[0] * self.BLOCK_SIZE, head[1] + direction[1] * self.BLOCK_SIZE)
        
        # Check if the new head position is equal to the food position
        if new_head == self.food_position:
            self.eat_food()
        else:
            # Check if the new head position is equal to any body position
            if new_head in self.positions[:-1]:
                self.reset()
            else:
                # Add the new head position to the body
                self.positions.append(new_head)


    def random_food_position(self):
        """
        Randomly generate a new food position, but don't place it on the snake.
        :return: None, Change the food position
        """
        # Generate a random food position
        food_x = random.randint(0, self.SCREEN_WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE * self.BLOCK_SIZE
        food_y = random.randint(0, self.SCREEN_HEIGHT - self.BLOCK_SIZE) // self.BLOCK_SIZE * self.BLOCK_SIZE
        
        # Check if the food position is on the snake
        while (food_x, food_y) in self.positions:
            food_x = random.randint(0, self.SCREEN_WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE * self.BLOCK_SIZE
            food_y = random.randint(0, self.SCREEN_HEIGHT - self.BLOCK_SIZE) // self.BLOCK_SIZE * self.BLOCK_SIZE
        
        self.food_position = (food_x, food_y)


    def reset(self):
        """
        Reset the snake to its initial state. Set the length to 1, the snake head position to ((SCREEN_WIDTH/2), (SCREEN_HEIGHT/2)), the score to 0, and randomly generate new food position.
        :return: None
        >>> snake = Snake(100, 100, 1, (51, 51))
        >>> snake.reset()
        self.length = 1
        self.positions = [(50, 50)]
        self.score = 0
        self.random_food_position()
        """
        self.length = 1
        self.positions = [((self.SCREEN_WIDTH / 2), (self.SCREEN_HEIGHT / 2))]
        self.score = 0
        self.random_food_position()


    def eat_food(self):
        """
        Increase the length of the snake by 1 and increase the score by 100. Randomly generate a new food position, but
        don't place it on the snake.
        :return: None
        >>> snake = Snake(100, 100, 1, (51, 51))
        >>> snake.move((1,1))
        >>> snake.eat_food()
        self.length = 2
        self.score = 10
        """
        self.length += 1
        self.score += 100
        self.random_food_position()


if __name__ == "__main__":
    # Test case
    snake = Snake(100, 100, 1, (51, 51))
    print("Initial snake state:")
    print("Length:", snake.length)
    print("Positions:", snake.positions)
    print("Score:", snake.score)
    print("Food position:", snake.food_position)

    snake.move((1,1))
    print("\nAfter moving the snake:")
    print("Length:", snake.length)
    print("Positions:", snake.positions)
    print("Score:", snake.score)
    print("Food position:", snake.food_position)

    snake.eat_food()
    print("\nAfter eating food:")
    print("Length:", snake.length)
    print("Positions:", snake.positions)
    print("Score:", snake.score)
    print("Food position:", snake.food_position)

    snake.reset()
    print("\nAfter resetting the snake:")
    print("Length:", snake.length)
    print("Positions:", snake.positions)
    print("Score:", snake.score)
    print("Food position:", snake.food_position)