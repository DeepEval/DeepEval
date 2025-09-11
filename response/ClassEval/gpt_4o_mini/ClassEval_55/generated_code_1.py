class Manacher:
    """
    This is a class that implements a Manacher algorithm to find the longest palindromic substring in a given string.
    """

    def __init__(self, input_string) -> None:
        """
        Initializes the Manacher class with the given input_string.
        :param input_string: The input_string to be searched, str.
        """
        self.input_string = input_string
        self.processed_string = self.preprocess_string(input_string)

    def preprocess_string(self, string):
        """
        Preprocess the input string to insert boundaries between characters for easier palindrome detection.
        """
        return '|' + '|'.join(string) + '|'

    def palindromic_length(self, center, diff, string):
        """
        Recursively calculates the length of the palindromic substring based on a given center, difference value, and input string.
        :param center: The center of the palindromic substring, int.
        :param diff: The difference between the center and the current position, int.
        :param string: The string to be searched, str.
        :return: The length of the palindromic substring, int.
        >>> manacher = Manacher('ababa')
        >>> manacher.palindromic_length(2, 1, 'a|b|a|b|a')
        2
        """
        left = center - diff
        right = center + diff
        length = 0
        
        while left >= 0 and right < len(string):
            if string[left] == string[right]:
                length += 1
                left -= 1
                right += 1
            else:
                break
                
        return length

    def palindromic_string(self):
        """
        Finds the longest palindromic substring in the given string.
        :return: The longest palindromic substring, str.
        >>> manacher = Manacher('ababaxse')
        >>> manacher.palindromic_string()
        'ababa'
        """
        max_len = 0
        max_center = 0
        processed_length = len(self.processed_string)
        pal_lengths = [0] * processed_length
        
        center = 0
        right_bound = 0
        
        for i in range(processed_length):
            mirror = 2 * center - i
            if i < right_bound:
                pal_lengths[i] = min(right_bound - i, pal_lengths[mirror])
                
            # Expand around center i
            current_length = self.palindromic_length(i, pal_lengths[i], self.processed_string)
            pal_lengths[i] += current_length
            
            # Update center and right_bound
            if i + pal_lengths[i] > right_bound:
                center = i
                right_bound = i + pal_lengths[i]

            # Update max_len and max_center
            if pal_lengths[i] > max_len:
                max_len = pal_lengths[i]
                max_center = i

        # Extract the longest palindromic substring
        start = (max_center - max_len) // 2
        return self.input_string[start:start + max_len]

# Testing the class and methods
if __name__ == "__main__":
    # Test for palindromic_length
    manacher = Manacher('ababa')
    length_output = manacher.palindromic_length(2, 1, 'a|b|a|b|a')
    print(f"Length of palindromic substring: {length_output}")  # Expected output: 2

    # Test for palindromic_string
    manacher2 = Manacher('ababaxse')
    longest_palindrome_output = manacher2.palindromic_string()
    print(f"Longest palindromic substring: '{longest_palindrome_output}'")  # Expected output: 'ababa'