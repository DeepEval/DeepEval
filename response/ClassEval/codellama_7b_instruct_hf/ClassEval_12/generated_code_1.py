import random

class BlackjackGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        deck = []
        for suit in ["Hearts", "Diamonds", "Spades", "Clubs"]:
            for rank in ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
  "Eight", "Nine", "Ten", "Jack", "Queen", "King"]:
                deck.append(f"{rank} of {suit}")
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        hand_value = 0
        for card in hand:
            rank = card[-1]
            if rank in ["Ace", "King", "Queen"]:
                hand_value += 10
            elif rank in ["Jack", "Ten"]:
                hand_value += 10
            else:
                hand_value += int(rank)
        return hand_value

    def check_winner(self, player_hand, dealer_hand):
        player_hand_value = self.calculate_hand_value(player_hand)
        dealer_hand_value = self.calculate_hand_value(dealer_hand)
        if player_hand_value > 21:
            return "Dealer wins"
        elif dealer_hand_value > 21:
            return "Player wins"
        elif player_hand_value == dealer_hand_value:
            return "Tie"
        else:
            if player_hand_value > dealer_hand_value:
                return "Player wins"
            else:
                return "Dealer wins"

if __name__ == "__main__":
    game = BlackjackGame()
    player_hand = game.create_deck()[:5]
    dealer_hand = game.create_deck()[5:10]
    print(f"Player''s hand: {player_hand}")
    print(f"Dealer''s hand: {dealer_hand}")
    print(f"Result: {game.check_winner(player_hand, dealer_hand)}")