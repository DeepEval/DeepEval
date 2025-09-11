import random

class TwentyFourPointGame:
    """
    This ia a game of twenty-four points, which provides to generate four numbers and check whether player's expression is equal to 24.
    """

    def __init__(self) -> None:
        self.nums = []


    def _generate_cards(self):
        """
        Generate random numbers between 1 and 9 for the cards.
        """
        self.nums = [random.randint(1, 9) for _ in range(4)]

    def get_my_cards(self):
        """
        Get a list of four random numbers between 1 and 9 representing the player's cards.
        :return: list of integers, representing the player's cards
        >>> game = TwentyFourPointGame()
        >>> game.get_my_cards()
        [7, 2, 8, 5]
        """
        self._generate_cards()
        return self.nums

    def answer(self, expression):
        """
        Check if a given mathematical expression using the cards can evaluate to 24.
        :param expression: string, mathematical expression using the cards
        :return: bool, True if the expression evaluates to 24, False otherwise
        >>> game = TwentyFourPointGame()
        >>> game.nums = [4, 3, 6, 6]
        >>> ans = "4*3+6+6"
        >>> ret = game.answer(ans)
        True
        """
        try:
            result = eval(expression)
            return result == 24
        except:
            return False

    def evaluate_expression(self, expression):
        """
        Evaluate a mathematical expression and check if the result is 24.
        :param expression: string, mathematical expression
        :return: bool, True if the expression evaluates to 24, False otherwise
        >>> game = TwentyFourPointGame()
        >>> nums = [4, 3, 6, 6]
        >>> ans = "4*3+6+6"
        >>> ret = game.evaluate_expression(ans)
        True
        """
        try:
            result = eval(expression)
            return result == 24
        except:
            return False

if __name__ == "__main__":
    game = TwentyFourPointGame()

    # Test case for get_my_cards
    cards = game.get_my_cards()
    print(f"Your cards are: {cards}")

    # Test case for answer
    expression = "4*3+6+6"
    game.nums = [4, 3, 6, 6]
    result = game.answer(expression)
    print(f"Does '{expression}' equal 24? {result}")

    # Test case for evaluate_expression
    expression = "4*3+6+6"
    result = game.evaluate_expression(expression)
    print(f"Does '{expression}' equal 24? {result}")