class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        """
        Initialize the score to be played
        :param text: str, score to be played
        """
        self.play_text = text

    def interpret(self, display=False):
        """
        Interpret the music score to be played
        :param display: Bool, representing whether to print the interpreted score
        :return: list of dict, The dict includes two fields, Chord and Tune, which are letters and numbers, respectively. If the input is empty or contains only whitespace, an empty list is returned.
        """
        if not self.play_text.strip():
            return []
        
        # Split the text into individual chords and their tunes
        chords_and_tunes = self.play_text.split()
        result = []
        
        for chord_tune in chords_and_tunes:
            chord = chord_tune[:-len(chord_tune.lstrip("0123456789"))]
            tune = chord_tune[len(chord):]
            result.append({"Chord": chord, "Tune": tune})
        
        if display:
            for item in result:
                self.display(item["Chord"], item["Tune"])
        
        return result

    def display(self, key, value):
        """
        Print out chord and play tune with following format: Normal Guitar Playing -- Chord: %s, Play Tune: %s
        :param key: str, chord
        :param value: str, play tune
        :return: str
        """
        print(f"Normal Guitar Playing -- Chord: {key}, Play Tune: {value}")

# Test cases
if __name__ == "__main__":
    context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
    play_list = context.interpret(display=False)
    print(play_list)  # Expected output: [{'Chord': 'C', 'Tune': '53231323'}, {'Chord': 'Em', 'Tune': '43231323'}, {'Chord': 'F', 'Tune': '43231323'}, {'Chord': 'G', 'Tune': '63231323'}]
    
    context = AutomaticGuitarSimulator("")
    play_list = context.interpret(display=False)
    print(play_list)  # Expected output: []
    
    context = AutomaticGuitarSimulator("   ")
    play_list = context.interpret(display=False)
    print(play_list)  # Expected output: []
    
    context = AutomaticGuitarSimulator("C53231323")
    context.display("C", "53231323")  # Expected output: Normal Guitar Playing -- Chord: C, Play Tune: 53231323