import random

class TwentyFourPointGame:
    """
    This is a game of twenty-four points, which provides to generate four numbers
    and check whether player's expression is equal to 24.
    """

    def __init__(self) -> None:
        self.nums = []
        self._generate_cards()

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
        >>> cards = game.get_my_cards()
        >>> len(cards) == 4
        True
        """
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
        return self.evaluate_expression(expression)

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
            return eval(expression) == 24
        except Exception as e:
            return False  # In case of any error in expression evaluation


if __name__ == "__main__":
    # Test case for get_my_cards
    game = TwentyFourPointGame()
    cards = game.get_my_cards()
    print("Generated cards:", cards)  # Should print 4 random numbers between 1 and 9
    print("Test get_my_cards:", len(cards) == 4)  # Should return True

    # Test case for answer method
    game.nums = [4, 3, 6, 6]
    expression = "4*3+6+6"
    is_correct = game.answer(expression)
    print("Test answer:", is_correct)  # Should return True

    # Test case for evaluate_expression method
    expression_eval = "4*3+6+6"
    eval_result = game.evaluate_expression(expression_eval)
    print("Test evaluate_expression:", eval_result)  # Should return True