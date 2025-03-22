def method(note_string):
    # Define the mapping of notes to their respective beat counts
    note_to_beats = {
        'o': 4,    # Whole note
        'o|': 2,   # Half note
        '.|': 1    # Quarter note
    }
    
    # Initialize the output list to store the beat counts
    output = []
    
    # Initialize a variable to hold the current note
    current_note = ''
    
    # Iterate through each character in the input string
    for char in note_string:
        # Append the character to the current note
        current_note += char
        # If the current note is in the mapping, it means we have a complete note
        if current_note in note_to_beats:
            # Append the beat count of the complete note to the output list
            output.append(note_to_beats[current_note])
            # Reset current note for the next set of characters
            current_note = ''
    
    # Return the output list containing the beat counts
    return output

# Test case for validation
test_input = "o.o|.|o|o"
expected_output = [4, 4, 1, 2, 2, 4]
# assert method(test_input) == expected_output, "Test case failed!"
method(test_input)

# If no assertion error is raised, the test case passes
# print("Test case passed!")