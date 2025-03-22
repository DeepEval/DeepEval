import re

def method(notes):
    """
    Input to this function is a string representing musical notes in a special ASCII format. Your task is to parse this string and return list of integers corresponding to how many beats does each not last. Here is a legend: 'o' - whole note, lasts four beats 'o|' - half note, lasts two beats '.|' - quater note, lasts one beat

    Args:
      notes (str): string representing musical notes in a special ASCII format

    Returns:
      list: list of integers corresponding to how many beats does each note last
    """

    notes_pattern = re.compile(r"[o|o\|.\|]")
    if not notes_pattern.fullmatch(notes):
        raise ValueError("Invalid musical notes format")

    beat_map = {"o": 4, "o|": 2, ".|": 1}
    beats = []
    for note in notes:
        beats.append(beat_map[note])

    return beats

# Test case
notes = "o|o.|.|o"
expected_output = [2, 1, 1, 4]
actual_output = method(notes)
# assert actual_output == expected_output, f"Expected {expected_output}, but got {actual_output}"

# print("Test case passed")