class Manacher:
    """
    This is a class that implements a manacher algorithm to find the longest palindromic substring in a given string.
    """

    def __init__(self, input_string) -> None:
        """
        Initializes the Manacher class with the given input_string.
        :param input_string: The input_string to be searched, str.
        """
        self.input_string = input_string

    def preprocess(self, s):
        """ 
        Preprocess the string to insert separators.
        :param s: The original string.
        :return: The processed string with separators.
        """
        return '|' + '|'.join(s) + '|'

    def palindromic_length(self, center, diff, string):
        """
        Calculates the length of the palindromic substring based on a given center, difference value, and input string.
        :param center: The center of the palindromic substring, int.
        :param diff: The difference between the center and the current position, int.
        :param string: The string to be searched, str.
        :return: The length of the palindromic substring, int.
        """
        length = 0
        while (center - diff - length >= 0 and
               center + diff + length < len(string) and
               string[center - diff - length] == string[center + diff + length]):
            length += 1
        return length - 1

    def palindromic_string(self):
        """
        Finds the longest palindromic substring in the given string.
        :return: The longest palindromic substring, str.
        """
        processed = self.preprocess(self.input_string)
        n = len(processed)
        P = [0] * n
        C = 0  # Center of the current palindrome
        R = 0  # Right edge of the current palindrome

        for i in range(n):
            mirror = 2 * C - i
            if i < R:
                P[i] = min(R - i, P[mirror])

            P[i] = self.palindromic_length(i, 1, processed)
            
            if i + P[i] > R:
                C = i
                R = i + P[i]

        max_len = max(P)
        center_index = P.index(max_len)
        start = (center_index - max_len) // 2
        return self.input_string[start:start + max_len]

if __name__ == "__main__":
    # Test case for palindromic_length
    manacher = Manacher('ababa')
    processed_string = manacher.preprocess('ababa')
    output_length = manacher.palindromic_length(2, 1, processed_string)
    print(f"Palindromic Length Test Output: {output_length}")

    # Test case for palindromic_string
    manacher = Manacher('ababaxse')
    output_string = manacher.palindromic_string()
    print(f"Palindromic String Test Output: {output_string}")