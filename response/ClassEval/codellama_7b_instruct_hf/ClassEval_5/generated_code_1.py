import re

class AutomaticGuitarSimulator:
    def __init__(self, text):
        self.play_text = text

    def interpret(self, display=False):
        chords = []
        for line in self.play_text.splitlines():
            chord, tune = line.split(" ")
            chords.append({"Chord": chord, "Tune": tune})
        if display:
            print(chords)
        return chords

    def display(self, key, value):
        return f"Normal Guitar Playing -- Chord: {key}, Play Tune: {value}"

if __name__ == "__main__":
    play_text = "C E\nG B\nD F#"
    simulator = AutomaticGuitarSimulator(play_text)
    
    chords = simulator.interpret(display=True)
    
    for chord in chords:
        print(simulator.display(chord["Chord"], chord["Tune"]))