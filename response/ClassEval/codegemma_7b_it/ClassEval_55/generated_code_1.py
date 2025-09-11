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
        >>> manacher = Manacher('ababa')
        >>> manacher.palindromic_length(2, 1, 'a|b|a|b|a')
        2

        """
        if center - diff < 0 or center + diff >= len(string):
            return diff
        if string[center - diff] != string[center + diff]:
            return diff - 1
        return self.palindromic_length(center, diff + 1, string)

    def palindromic_string(self):
        """
        Finds the longest palindromic substring in the given string.
        :return: The longest palindromic substring, str.
        >>> manacher = Manacher('ababaxse')
        >>> manacher.palindromic_string()
        'ababa'

        """
        if not self.input_string:
            return ""
        
        string = self.input_string
        string = '|'.join(list(string))
        n = len(string)
        palindrome_length = [0] * n
        palindrome_length[0] = 0
        palindrome_length[1] = 1
        
        for i in range(2, n):
            if palindrome_length[i - 1] < palindrome_length[i - 2] + 2:
                palindrome_length[i] = palindrome_length[i - 2] + 2
            else:
                palindrome_length[i] = palindrome_length[i - 1]
            
            if i + palindrome_length[i] - 1 >= n or i - palindrome_length[i] + 1 <= 0:
                continue
            
            if string[i - palindrome_length[i] + 1] == string[i + palindrome_length[i] - 1]:
                palindrome_length[i] += 1
        
        max_length = max(palindrome_length)
        for i in range(n - max_length + 1):
            if palindrome_length[i] == max_length:
                return self.input_string[i:i + max_length - 1]

if __name__ == "__main__":
    manacher = Manacher('ababa')
    print(manacher.palindromic_length(2, 1, 'a|b|a|b|a'))
    print(manacher.palindromic_string())