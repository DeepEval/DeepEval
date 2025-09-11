


import random

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
        self.nums = random.sample(range(1, 10), 4)


    def get_my_cards(self):
        """
        Get a list of four random numbers between 1 and 9 representing the player's cards.
        :return: list of integers, representing the player's cards
        >>> game = TwentyFourPointGame()
        >>> game.get_my_cards()

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
            result = eval(expression)
            return result == 24
        except:
            return False

if __name__ == "__main__":

    game = TwentyFourPointGame()
    game._generate_cards()

    cards = game.get_my_cards()
    print(f"生成的卡片: {cards}")

    expressions = [
        f"({cards[0]} + {cards[1]}) * ({cards[2]} + {cards[3]})",
        f"({cards[0]} * {cards[1]}) + ({cards[2]} * {cards[3]})",
        f"{cards[0]} * {cards[1]} * {cards[2]} * {cards[3]}",
        f"({cards[0]} + {cards[1]} + {cards[2]} + {cards[3]})",
        f"{cards[0]} * {cards[1]} - {cards[2]} - {cards[3]}"
    ]
    
    print("\n测试表达式:")
    for expr in expressions:
        result = game.answer(expr)
        print(f"表达式: {expr} = 24? {result}")

    test_cards = [4, 3, 6, 6]
    test_expr = "4*3+6+6"
    game.nums = test_cards
    test_result = game.answer(test_expr)
    print(f"\n控制测试: 表达式 '{test_expr}' 使用卡片 {test_cards} = 24? {test_result}")