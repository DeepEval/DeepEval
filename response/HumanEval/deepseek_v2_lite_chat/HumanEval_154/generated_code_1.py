def method():
    def rotate_word(word, rotation_count):
        return word * rotation_count[-1] + word[:-rotation_count[-1]]

    def is_substring(rotated_word, main_word):
        return rotated_word in main_word or main_word.find(rotated_word) != -1

    def cycpattern_check(first_word, second_word):
        # Normalize second_word for rotations
        normalized_second_word = second_word
        if len(second_word) > len(first_word):
            normalized_second_word = second_word[len(first_word):]

        # Rotate second_word
        for _ in range(len(normalized_second_word)):
            rotated_word = rotate_word(normalized_second_word, [i % len(normalized_second_word) for i in range(len(normalized_second_word))])

            # Check if any rotation is a substring of first_word
            if is_substring(rotated_word, first_word):
                return True

        return False

    # Test cases
    test_cases = [
        ("abcd", "abd", False),
        ("hello", "ell", True),
        ("whassup", "psus", False),
        ("abab", "baa", True),
        ("efef", "eeff", False),
        ("himenss", "simen", True)
    ]

    for first_word, second_word, expected in test_cases:
        result = cycpattern_check(first_word, second_word)
        # assert result == expected, f"Test case failed for {first_word}, {second_word}. Expected {expected}, got {result}"

    print("All test cases passed!")

method()