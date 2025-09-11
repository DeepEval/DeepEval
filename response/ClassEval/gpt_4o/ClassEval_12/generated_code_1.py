import random

class BlackjackGame:
    """
    This is a class representing a game of blackjack, which includes creating a deck, calculating the value of a hand, and determining the winner based on the hand values of the player and dealer.
    """

    def __init__(self):
        """
        Initialize the Blackjack Game with the attribute deck, player_hand and dealer_hand.
        While initializing deck attribute, call the create_deck method to generate.
        The deck stores 52 random order poker with the Jokers removed, format is ['AS', '2S', ...].
        player_hand is a list which stores player's hand cards.
        dealer_hand is is a list which stores dealer's hand cards.
        """
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        """
        Create a deck of 52 cards, which stores 52 random order poker with the Jokers removed.
        :return: a list of 52 random order poker with the Jokers removed, format is ['AS', '2S', ...].
        """
        suits = ['S', 'H', 'D', 'C']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [rank + suit for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        """
        Calculate the value of the poker cards stored in hand list according to the rules of the Blackjack Game.
        :param hand: list
        :return: the value of the poker cards stored in hand list, a number.
        """
        value = 0
        num_aces = 0
        for card in hand:
            rank = card[:-1]
            if rank.isdigit():
                value += int(rank)
            elif rank in ['J', 'Q', 'K']:
                value += 10
            elif rank == 'A':
                num_aces += 1
                value += 11
        
        while value > 21 and num_aces:
            value -= 10
            num_aces -= 1
        
        return value

    def check_winner(self, player_hand, dealer_hand):
        """
        Determines the winner of a game by comparing the hand values of the player and dealer.
        :param player_hand: list
        :param dealer_hand: list
        :return: the result of the game, only two certain str: 'Dealer wins' or 'Player wins'
        """
        player_value = self.calculate_hand_value(player_hand)
        dealer_value = self.calculate_hand_value(dealer_hand)
        
        if player_value <= 21 and (player_value > dealer_value or dealer_value > 21):
            return 'Player wins'
        else:
            return 'Dealer wins'

if __name__ == "__main__":
    black_jack_game = BlackjackGame()
    
    # Test case for create_deck method
    deck = black_jack_game.create_deck()
    print("Deck:", deck)
    print("Deck length:", len(deck))  # Should be 52

    # Test case for calculate_hand_value method
    hand_value = black_jack_game.calculate_hand_value(['QD', '9D', 'JC', 'QH', 'AS'])
    print("Hand Value:", hand_value)  # Should be 40

    # Test case for check_winner method
    result = black_jack_game.check_winner(['QD', '9D', 'JC', 'QH', 'AS'], ['QD', '9D', 'JC', 'QH', '2S'])
    print("Winner:", result)  # Should be 'Player wins'