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

    def palindromic_length(self, center, diff, string):
        """
        Recursively calculates the length of the palindromic substring based on a given center, difference value, and input string.
        :param center: The center of the palindromic substring, int.
        :param diff: The difference between the center and the current position, int.
        :param string: The string to be searched, str.
        :return: The length of the palindromic substring, int.
        """
        # Base case: If the center is negative or out of bounds, return 0
        if center < 0 or center >= len(string):
            return 0
        
        # If the difference is 0, it means we've found the center of the palindrome
        if diff == 0:
            return 1
        
        # If the characters at the current center and the given difference are the same, 
        # recursively calculate the length of the palindrome on the left side
        if string[center] == string[center - diff]:
            return 2 + self.palindromic_length(center - diff, diff - 1, string)
        
        # If the characters are different, return 1 (a single character is a palindrome)
        return 1

    def palindromic_string(self):
        """
        Finds the longest palindromic substring in the given string.
        :return: The longest palindromic substring, str.
        """
        # Preprocess the input string to handle even length palindromes
        string ='' + self.input_string + '#'
        
        # Initialize the maximum length and center of the palindrome
        max_len = 0
        max_center = 0
        
        # Iterate over the preprocessed string to find the maximum length palindrome
        for i in range(len(string)):
            if i < len(string) / 2:
                mirror = len(string) - i - 1
                diff = 1
            else:
                mirror = 2 * i - len(string) - 1
                diff = i - len(string) // 2
            
            # Calculate the length of the palindrome centered at the current position
            length = self.palindromic_length(i, diff, string)
            
            # Update the maximum length and center if the current palindrome is longer
            if length > max_len:
                max_len = length
                max_center = i - length // 2
        
        # Return the longest palindromic substring
        return self.input_string[max_center - max_len // 2 : max_center + max_len // 2]

if __name__ == "__main__":
    manacher = Manacher('ababa')
    # Test case for palindromic_length
    output = manacher.palindromic_length(2, 1, 'a|b|a|b|a')
    print("palindromic_length:", output)  # Output: 2
    
    # Test case for palindromic_string
    output = manacher.palindromic_string()
    print("palindromic_string:", output)  # Output: 'ababa'