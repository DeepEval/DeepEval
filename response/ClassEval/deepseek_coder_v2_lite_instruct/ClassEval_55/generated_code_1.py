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

    def preprocess(self, s):
        if len(s) == 0: return "^$"
        ret = "^"
        for i in range(len(s)):
            ret += "#" + s[i]
        ret += "#$"
        return ret

    def palindromic_length(self, center, diff, string):
        """
        Recursively calculates the length of the palindromic substring based on a given center, difference value, and input string.
        :param center: The center of the palindromic substring, int.
        :param diff: The difference between the center and the current position, int.
        :param string: The string to be searched, str.
        :return: The length of the palindromic substring, int.
        """
        left = center - (1 + diff)
        right = center + (1 + diff)
        while (left >= 0 and right < len(string) and string[left] == string[right]):
            left -= 1
            right += 1
        return right - left - 1

    def palindromic_string(self):
        """
        Finds the longest palindromic substring in the given string.
        :return: The longest palindromic substring, str.
        """
        T = self.preprocess(self.input_string)
        n = len(T)
        P = [0] * n
        C = R = 0
        for i in range (1, n-1):
            i_mirror = 2*C - i  # i' = C - (i-C)
            if R > i:
                P[i] = min(R - i, P[i_mirror])
            # Attempt to expand palindrome centered at current position
            while T[i + 1 + P[i]] == T[i - 1 - P[i]]:
                P[i] += 1
            # If palindrome centered at current position expand past R, adjust center and R
            if i + P[i] > R:
                C = i
                R = i + P[i]
        # Find the maximum element in P
        max_len = 0
        center_index = 0
        for i in range(1, n-1):
            if P[i] > max_len:
                max_len = P[i]
                center_index = i
        start = (center_index - max_len) // 2
        return self.input_string[start:start + max_len]

if __name__ == "__main__":
    manacher = Manacher('ababaxse')
    print(manacher.palindromic_string())  # Expected output: 'ababa'