class Manacher:
    """
    This is a class that implements a manacher algorithm to find the Longest palindromic substring in a given string.
    """

    def __init__(self, input_string) -> None:
        """
        Initializes the Manacher class with the given input_string.
        :param input_string: The input_string to be searched, str.
        """
        self.input_string = input_string
        self.processed_string = self._preprocess_string(input_string)
        self.palindrome_radii = [0] * len(self.processed_string)
        self.center = 0
        self.right = 0

    def _preprocess_string(self, input_string):
        """
        Preprocesses the input string by inserting special characters to handle odd and even length palindromes.
        :param input_string: The input string, str.
        :return: The preprocessed string, str.
        """
        processed_string = '#' + '#'.join(input_string) + '#'
        return processed_string

    def palindromic_length(self, center, diff, string):
        """
        Recursively calculates the length of the palindromic substring based on a given center, difference value, and input string.
        :param center: The center of the palindromic substring, int.
        :param diff: The difference between the center and the current position, int.
        :param string: The string to be searched, str.
        :return: The length of the palindromic substring, int.
        """
        length = 0
        while center - diff - length >= 0 and center + diff + length < len(string) and string[center - diff - length] == string[center + diff + length]:
            length += 1
        return length

    def palindromic_string(self):
        """
        Finds the longest palindromic substring in the given string.
        :return: The longest palindromic substring, str.
        """
        for i in range(len(self.processed_string)):
            if i < self.right:
                mirror = 2 * self.center - i
                self.palindrome_radii[i] = min(self.right - i, self.palindrome_radii[mirror])
            
            while i - self.palindrome_radii[i] - 1 >= 0 and i + self.palindrome_radii[i] + 1 < len(self.processed_string) and self.processed_string[i - self.palindrome_radii[i] - 1] == self.processed_string[i + self.palindrome_radii[i] + 1]:
                self.palindrome_radii[i] += 1

            if i + self.palindrome_radii[i] > self.right:
                self.center = i
                self.right = i + self.palindrome_radii[i]

        max_length = max(self.palindrome_radii)
        center_index = self.palindrome_radii.index(max_length)
        start = (center_index - max_length) // 2
        end = start + max_length
        return self.input_string[start:end]



    

if __name__ == "__main__":
    instance = Manacher('ababaxse')
    print(instance.palindromic_string())