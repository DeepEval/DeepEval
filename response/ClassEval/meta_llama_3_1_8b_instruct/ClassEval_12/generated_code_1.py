import random

class BlackjackGame:
    """
    This is a class representing a game of blackjack, which includes creating a deck, 
    calculating the value of a hand, and determine the winner based on the hand values of 
    the player and dealer.
    """

    def __init__(self):
        """
        Initialize the Blackjack Game with the attribute deck, player_hand and dealer_hand.
        While initializing deck attribute, call the create_deck method to generate.
        The deck stores 52 rondom order poker with the Jokers removed, format is ['AS', '2S',...].
        player_hand is a list which stores player's hand cards.
        dealer_hand is is a list which stores dealer's hand cards.
        """
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        """
        Create a deck of 52 cards, which stores 52 rondom order poker with the Jokers removed.
        :return: a list of 52 rondom order poker with the Jokers removed, format is ['AS', '2S',...].
        """
        suits = ['S', 'H', 'D', 'C']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        deck = [rank + suit for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        """
        Calculate the value of the poker cards stored in hand list according to the rules of the Blackjack Game.
        If the card is a digit, its value is added to the total hand value.
        Value of J, Q, or K is 10, while Aces are worth 11.
        If the total hand value exceeds 21 and there are Aces present, one Ace is treated as having a value of 1 instead of 11,
        until the hand value is less than or equal to 21, or all Aces have been counted as value of 1.
        :param hand: list
        :return: the value of the poker cards stored in hand list, a number.
        """
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}
        total_value = sum(values[card[0]] for card in hand)
        aces = sum(1 for card in hand if card[0] == 'A')
        while total_value > 21 and aces:
            total_value -= 10
            aces -= 1
        return total_value

    def check_winner(self, player_hand, dealer_hand):
        """
        Determines the winner of a game by comparing the hand values of the player and dealer.
        rule:
        If both players have hand values that are equal to or less than 21, the winner is the one whose hand value is closer to 21.
        Otherwise, the winner is the one with the lower hand value.
        :param player_hand: list
        :param dealer_hand: list
        :return: the result of the game, only two certain str: 'Dealer wins' or 'Player wins'
        """
        player_value = self.calculate_hand_value(player_hand)
        dealer_value = self.calculate_hand_value(dealer_hand)
        if player_value <= 21 and dealer_value <= 21:
            if player_value > dealer_value:
                return 'Player wins'
            elif player_value < dealer_value:
                return 'Dealer wins'
            else:
                return 'Tie'
        elif player_value <= 21 and dealer_value > 21:
            return 'Player wins'
        elif player_value > 21 and dealer_value <= 21:
            return 'Dealer wins'
        else:
            return 'Dealer wins'

if __name__ == "__main__":
    black_jack_game = BlackjackGame()
    print("Deck:")
    print(black_jack_game.create_deck())
    print("\nPlayer's Hand Value:")
    print(black_jack_game.calculate_hand_value(['QD', '9D', 'JC', 'QH', 'AS']))
    print("\nCheck Winner:")
    print(black_jack_game.check_winner(['QD', '9D', 'JC', 'QH', 'AS'], ['QD', '9D', 'JC', 'QH', '2S']))