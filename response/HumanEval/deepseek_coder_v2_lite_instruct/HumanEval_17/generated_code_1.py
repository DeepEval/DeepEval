def method():
    # Define the mapping from note representation to beats
    note_to_beats = {
        'o': 4,
        'o|': 2,
        '.|': 1
    }
    
    # Example input string
    input_string = "o.o|.o.o|o|"
    
    # Initialize the output list
    output = []
    
    # Iterate through the input string and map each note to its corresponding beats
    i = 0
    while i < len(input_string):
        found = False   
        for note, beats in note_to_beats.items():
            # if input_string[i:].startswith(note):
            if input_string[i:i+len(note)] == note:  # Check from current position i
                output.append(beats)
                i += len(note)
                found = True
                break
        if not found:
            i += 1  # Move to the next character if no match is found
    
    return output

# Test case
print(method())  # Expected output: [4, 1, 4, 1, 2, 4]