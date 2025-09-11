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
        head_x = self.positions[0][0] + direction[0] * self.BLOCK_SIZE
        head_y = self.positions[0][1] + direction[1] * self.BLOCK_SIZE

        if head_x < 0 or head_x >= self.SCREEN_WIDTH or head_y < 0 or head_y >= self.SCREEN_HEIGHT:
            raise Exception("Game Over: Snake hit the boundary")

        for body in self.positions[1:]:
            if head_x == body[0] and head_y == body[1]:
                raise Exception("Game Over: Snake hit itself")

        self.positions.insert(0, (head_x, head_y))

        if self.positions[0] == self.food_position:
            self.score += 100
            self.eat_food()
        else:
            self.positions.pop()


    def random_food_position(self):
        """
        Randomly generate a new food position, but don't place it on the snake.
        :return: None, Change the food position
        """
        while True:
            x = random.randint(0, (self.SCREEN_WIDTH // self.BLOCK_SIZE) - 1) * self.BLOCK_SIZE
            y = random.randint(0, (self.SCREEN_HEIGHT // self.BLOCK_SIZE) - 1) * self.BLOCK_SIZE
            if (x, y) not in self.positions:
                self.food_position = (x, y)
                break


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
    snake = Snake(100, 100, 10, (51, 51))

    # Test case for move
    try:
        snake.move((1, 0))
        print(f"Snake position: {snake.positions}")
        print(f"Snake score: {snake.score}")
    except Exception as e:
        print(e)

    # Test case for random_food_position
    snake.reset()
    print(f"New food position: {snake.food_position}")

    # Test case for reset
    snake.reset()
    print(f"Snake length after reset: {snake.length}")
    print(f"Snake position after reset: {snake.positions}")
    print(f"Snake score after reset: {snake.score}")

    # Test case for eat_food
    snake.move((1, 0))
    snake.eat_food()
    print(f"Snake length after eating food: {snake.length}")
    print(f"Snake score after eating food: {snake.score}")