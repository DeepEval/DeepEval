import re

def method(notes_string):
    # Define a regular expression to match the ASCII notes
    note_pattern = r'o|o\||.\|'
    
    # Use the re.findall method to find all occurrences of the pattern
    notes = re.findall(note_pattern, notes_string)
    
    # Flatten the list of notes into a single list of integers
    notes = [int(note[0]) for note in notes]
    
    return notes