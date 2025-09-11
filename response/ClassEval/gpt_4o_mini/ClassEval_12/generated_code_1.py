import random

class BlackjackGame:
    """
    This is a class representing a game of blackjack, which includes creating a deck, calculating the value of a hand, 
    and determining the winner based on the hand values of the player and dealer.
    """

    def __init__(self):
        """
        Initialize the Blackjack Game with the attribute deck, player_hand, and dealer_hand.
        While initializing the deck attribute, call the create_deck method to generate.
        The deck stores 52 random order poker with the Jokers removed.
        """
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        """
        Create a deck of 52 cards, which stores 52 random order poker with the Jokers removed.
        :return: a list of 52 random order poker with the Jokers removed, format is ['AS', '2S', ...].
        """
        suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
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
        aces_count = 0
        
        for card in hand:
            rank = card[:-1]
            if rank in ['J', 'Q', 'K']:
                value += 10
            elif rank == 'A':
                value += 11
                aces_count += 1
            else:
                value += int(rank)

        while value > 21 and aces_count:
            value -= 10
            aces_count -= 1

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

        if player_value > 21:
            return 'Dealer wins'
        elif dealer_value > 21:
            return 'Player wins'
        elif player_value > dealer_value:
            return 'Player wins'
        elif dealer_value > player_value:
            return 'Dealer wins'
        else:
            return 'Dealer wins'  # If it's a tie, dealer wins by default

if __name__ == "__main__":
    # Test the create_deck method
    game = BlackjackGame()
    deck_output = game.create_deck()
    print("Deck:", deck_output)
    
    # Test the calculate_hand_value method
    hand_value_output = game.calculate_hand_value(['QD', '9D', 'JC', 'QH', 'AS'])
    print("Hand Value:", hand_value_output)

    # Test the check_winner method
    winner_output = game.check_winner(['QD', '9D', 'JC', 'QH', 'AS'], ['QD', '9D', 'JC', 'QH', '2S'])
    print("Winner:", winner_output)