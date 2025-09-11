class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        """
        Initialize the score to be played
        :param text:str, score to be played
        """
        self.play_text = text

    def interpret(self, display=False):
        """
        Interpret the music score to be played
        :param display:Bool, representing whether to print the interpreted score
        :return: list of dict, The dict includes two fields, Chord and Tune, which are letters and numbers, respectively. If the input is empty or contains only whitespace, an empty list is returned.
        >>> context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
        >>> play_list = context.interpret(display = False)
        [{'Chord': 'C', 'Tune': '53231323'}, {'Chord': 'Em', 'Tune': '43231323'}, {'Chord': 'F', 'Tune': '43231323'}, {'Chord': 'G', 'Tune': '63231323'}]

        """
        if self.play_text.strip() == "":
            return []
        play_list = self.play_text.split(" ")
        result = []
        for item in play_list:
            chord = item.split(":")[0]
            tune = item.split(":")[1]
            result.append({"Chord": chord, "Tune": tune})
        return result

    def display(self, key, value):
        """
        Print out chord and play tune with following format: Normal Guitar Playing -- Chord: %s, Play Tune: %s
        :param key:str, chord
        :param value:str, play tune
        :return: str
        >>> context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
        >>> context.display("C", "53231323")
        Normal Guitar Playing -- Chord: C, Play Tune: 53231323

        """
        print(f"Normal Guitar Playing -- Chord: {key}, Play Tune: {value}")
        return f"Normal Guitar Playing -- Chord: {key}, Play Tune: {value}"

if __name__ == "__main__":
    instance = AutomaticGuitarSimulator("C:53231323 Em:43231323 F:43231323 G:63231323")
    output_interpret = instance.interpret(display=True)
    print(output_interpret)
    output_display = instance.display("C", "53231323")
    print(output_display)