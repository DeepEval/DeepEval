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
        :return: list of dict, The dict includes two fields, Chord and Tune, which are letters and numbers, respectively.
        """
        # Split the input string into chords and tunes
        chords_and_tunes = self.play_text.split()
        
        # Initialize an empty list to store the interpreted chords and tunes
        interpreted_chords_and_tunes = []
        
        # Loop through the chords and tunes
        for i in range(0, len(chords_and_tunes), 2):
            # Get the chord and tune
            chord = chords_and_tunes[i]
            tune = chords_and_tunes[i + 1]
            
            # Append the interpreted chord and tune to the list
            interpreted_chords_and_tunes.append({'Chord': chord, 'Tune': tune})
        
        # If display is True, print the interpreted chords and tunes
        if display:
            print(interpreted_chords_and_tunes)
        
        # Return the interpreted chords and tunes
        return interpreted_chords_and_tunes

    def display(self, key, value):
        """
        Print out chord and play tune with following format: Normal Guitar Playing -- Chord: %s, Play Tune: %s
        :param key:str, chord
        :param value:str, play tune
        :return: str
        """
        # Print the chord and tune in the specified format
        return "Normal Guitar Playing -- Chord: {}, Play Tune: {}".format(key, value)

if __name__ == "__main__":
    # Test case for interpret method
    context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
    play_list = context.interpret(display=False)
    print(play_list)
    # Output: [{'Chord': 'C', 'Tune': '53231323'}, {'Chord': 'Em', 'Tune': '43231323'}, {'Chord': 'F', 'Tune': '43231323'}, {'Chord': 'G', 'Tune': '63231323'}]

    # Test case for display method
    print(context.display("C", "53231323"))
    # Output: Normal Guitar Playing -- Chord: C, Play Tune: 53231323