def method(input_string):
    # Initialize an empty list to store the beats
    output = []
    
    # Define a mapping for note representations to their respective beats
    note_mapping = {
        'o': 4,   # whole note
        'o|': 2,  # half note
        '.|': 1   # quarter note
    }
    
    # Parse the input string
    i = 0
    while i < len(input_string):
        if input_string[i] == 'o':
            # Check if the next character is '|'
            if i + 1 < len(input_string) and input_string[i + 1] == '|':
                output.append(note_mapping['o|'])
                i += 2  # Skip the two characters
            else:
                output.append(note_mapping['o'])
                i += 1  # Skip the one character
        elif input_string[i] == '.':
            # The next character should be '|'
            if i + 1 < len(input_string) and input_string[i + 1] == '|':
                output.append(note_mapping['.|'])
                i += 2  # Skip the two characters
            else:
                i += 1  # Skip the character if unexpected
        else:
            i += 1  # Move to the next character if it doesn't match any note
    
    return output

# Test case for validation
input_string = "o o| .| o .| o|"
output = method(input_string)
print(output)  # Expected output: [4, 2, 1, 4, 1, 2]