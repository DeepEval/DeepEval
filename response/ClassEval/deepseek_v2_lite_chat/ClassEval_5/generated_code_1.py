class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        """
        Initialize the score to be played
        :param text: str, score to be played
        """
        self.play_text = text
        if not self.play_text.strip():  # Check for empty or whitespace-only input
            self.play_text = None

    def interpret(self, display=True):
        """
        Interpret the music score to be played
        :param display: Bool, representing whether to print the interpreted score
        :return: list of dict, The dict includes two fields, Chord and Tune, which are letters and numbers, respectively. If the input is empty or contains only whitespace, an empty list is returned.
        """
        if self.play_text is None:  # Check if the input is empty or whitespace
            return []

        # Split the play_text into lines
        lines = self.play_text.split('\n')
        result = []

        # Interpret each line
        for line in lines:
            if not line.strip():  # Skip empty lines
                continue
            try:
                # Extract chord and tune
                chord, tune = line.split(' ')
                result.append({'Chord': chord, 'Tune': tune})
            except ValueError:
                print(f"Error parsing line: {line}. Skipping.")
                continue

        if display:
            for item in result:
                print(f"Chord: {item['Chord']}, Play Tune: {item['Tune']}")

        return result

    def display(self, key, value):
        """
        Print out chord and play tune with following format: Normal Guitar Playing -- Chord: %s, Play Tune: %s
        :param key: str, chord
        :param value: str, play tune
        :return: str
        """
        print(f"Normal Guitar Playing -- Chord: {key}, Play Tune: {value}")

if __name__ == "__main__":

    # Test cases
    context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
    play_list = context.interpret(display=False)
    print(play_list)

    context.display("C", "53231323")