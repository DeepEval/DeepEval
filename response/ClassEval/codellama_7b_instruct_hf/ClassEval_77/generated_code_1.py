import random

class Snake:
    """
    The class is a snake game, with allows snake to move and eat food, and also enables to reset, and generate a random food position.
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
        """
        new_position = (self.positions[0][0] + direction[0] * self.BLOCK_SIZE, self.positions[0][1] + direction[1] * self.BLOCK_SIZE)
        if new_position == self.food_position:
            self.length += 1
            self.score += 100
            self.random_food_position()
        elif new_position in self.positions:
            self.reset()
        else:
            self.positions.insert(0, new_position)
            self.length += 1


    def random_food_position(self):
        """
        Randomly generate a new food position, but don't place it on the snake.
        :return: None, Change the food position
        """
        while True:
            x = random.randint(0, self.SCREEN_WIDTH // self.BLOCK_SIZE - 1)
            y = random.randint(0, self.SCREEN_HEIGHT // self.BLOCK_SIZE - 1)
            if (x, y) not in self.positions:
                self.food_position = (x * self.BLOCK_SIZE, y * self.BLOCK_SIZE)
                break


    def reset(self):
        """
        Reset the snake to its initial state. Set the length to 1, the snake head position to ((SCREEN_WIDTH/2), (SCREEN_HEIGHT/2)), the score to 0, and randomly generate new food position.
        :return: None
        """
        self.length = 1
        self.positions = [(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)]
        self.score = 0
        self.random_food_position()


    def eat_food(self):
        """
        Increase the length of the snake by 1 and increase the score by 100. Randomly generate a new food position, but
        don't place it on the snake.
        :return: None
        """
        self.length += 1
        self.score += 100
        self.random_food_position()

if __name__ == "__main__":

    # Test cases
    snake = Snake(100, 100, 1, (51, 51))

    # Test case 1: Move the snake in the positive x direction
    snake.move((1,0))
    print(snake.positions)

    # Test case 2: Move the snake in the positive y direction
    snake.move((0,1))
    print(snake.positions)

    # Test case 3: Move the snake in the negative x direction
    snake.move((-1,0))
    print(snake.positions)

    # Test case 4: Move the snake in the negative y direction
    snake.move((0,-1))
    print(snake.positions)

    # Test case 5: Move the snake to the position of the food
    snake.move((snake.food_position[0] - snake.positions[0][0], snake.food_position[1] - snake.positions[0][1]))
    print(snake.positions)

    # Test case 6: Move the snake to a position that is already occupied by its body
    snake.move((snake.positions[1][0], snake.positions[1][1]))
    print(snake.positions)

    # Test case 7: Move the snake to a position that is not occupied by its body
    snake.move((snake.positions[0][0] + 1, snake.positions[0][1]))
    print(snake.positions)

    # Test case 8: Eat the food
    snake.eat_food()
    print(snake.length, snake.score, snake.positions)

    # Test case 9: Reset the snake
    snake.reset()
    print(snake.length, snake.score, snake.positions)

    # Test case 10: Move the snake in the positive x direction after resetting
    snake.move((1,0))
    print(snake.positions)