import random
import operator

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
        self.nums = random.sample(range(1, 10), 4)

    def get_my_cards(self):
        """
        Get a list of four random numbers between 1 and 9 representing the player's cards.
        :return: list of integers, representing the player's cards
        >>> game = TwentyFourPointGame()
        >>> game.get_my_cards()
        [3, 5, 8, 2]
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
        ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        try:
            result = eval(expression, {'__builtins__': None}, {'nums': self.nums, '__import__': None})
            return result == 24
        except ZeroDivisionError:
            return False
        except Exception as e:
            return False


if __name__ == "__main__":
    game = TwentyFourPointGame()
    print(game.get_my_cards())  # Output: [8, 2, 5, 3]
    game.nums = [4, 3, 6, 6]
    print(game.answer("4*3+6+6"))  # Output: True
    print(game.evaluate_expression("4*3+6+6"))  # Output: True
    print(game.evaluate_expression("4/0+3+6+6"))  # Output: False