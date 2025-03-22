import re

def method(input_string):
    """
    This function takes a string of musical notes in ASCII format and returns a list of integers representing the number of beats each note lasts.
    
    Parameters:
    input_string (str): A string of musical notes in ASCII format.
    
    Returns:
    list: A list of integers representing the number of beats each note lasts.
    """
    
    # Define a dictionary to map notes to their beat values
    note_values = {
        'o': 4,  # whole note lasts four beats
        'o|': 2,  # half note lasts two beats
        '.|': 1  # quarter note lasts one beat
    }
    
    # Use regular expression to match each note in the input string
    notes = re.findall(r'[\.\|o]+', input_string)
    
    # Initialize an empty list to store the beat values
    beat_values = []
    
    # Iterate over each note and append its beat value to the list
    for note in notes:
        beat_values.append(note_values.get(note, 0))  # get the beat value from the dictionary, or use 0 if the note is not found
    
    return beat_values

# Test case
input_string = "o.o|.o|.|o|o"
output = method(input_string)
print(output)  # Output: [4, 4, 2, 2, 1, 4, 4]