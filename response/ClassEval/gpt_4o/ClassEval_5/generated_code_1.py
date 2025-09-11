class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        """
        Initialize the score to be played.
        :param text: str, score to be played
        """
        self.play_text = text

    def interpret(self, display=False):
        """
        Interpret the music score to be played.
        :param display: Bool, representing whether to print the interpreted score
        :return: list of dict, The dict includes two fields, Chord and Tune, which are letters and numbers, respectively.
                 If the input is empty or contains only whitespace, an empty list is returned.
        """
        if not self.play_text.strip():
            return []

        chords = self.play_text.split()
        result = []

        for chord in chords:
            chord_name = ''.join(filter(str.isalpha, chord))
            play_tune = ''.join(filter(str.isdigit, chord))
            result.append({'Chord': chord_name, 'Tune': play_tune})

        if display:
            for item in result:
                self.display(item['Chord'], item['Tune'])

        return result

    def display(self, key, value):
        """
        Print out chord and play tune with the following format: Normal Guitar Playing -- Chord: %s, Play Tune: %s
        :param key: str, chord
        :param value: str, play tune
        :return: str
        """
        output = f"Normal Guitar Playing -- Chord: {key}, Play Tune: {value}"
        print(output)
        return output


if __name__ == "__main__":
    # Test case for interpret method
    context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
    play_list = context.interpret(display=False)
    print(play_list)
    # Expected output:
    # [{'Chord': 'C', 'Tune': '53231323'}, {'Chord': 'Em', 'Tune': '43231323'}, {'Chord': 'F', 'Tune': '43231323'}, {'Chord': 'G', 'Tune': '63231323'}]

    # Test case for display method
    output = context.display("C", "53231323")
    # Expected output:
    # Normal Guitar Playing -- Chord: C, Play Tune: 53231323