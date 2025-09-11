import random
import re

class TwentyFourPointGame:
    """
    This is a game of twenty-four points, which provides to generate four numbers and check whether player's expression is equal to 24.
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
        """
        self._generate_cards()
        return self.nums

    def answer(self, expression):
        """
        Check if a given mathematical expression using the cards can evaluate to 24.
        :param expression: string, mathematical expression using the cards
        :return: bool, True if the expression evaluates to 24, False otherwise
        """
        try:
            result = self.evaluate_expression(expression)
            return result == 24
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return False

    def evaluate_expression(self, expression):
        """
        Evaluate a mathematical expression and check if the result is 24.
        :param expression: string, mathematical expression
        :return: bool, True if the expression evaluates to 24, False otherwise
        """
        try:
            # Evaluate the expression using Python's eval function
            result = eval(expression)
            return result
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None

if __name__ == "__main__":
    game = TwentyFourPointGame()
    
    # Test get_my_cards method
    print("Player's cards:", game.get_my_cards())
    
    # Example expression to test
    example_expression = "4*3+6+6"
    print(f"Expression: {example_expression}")
    print("Evaluates to 24:", game.answer(example_expression))