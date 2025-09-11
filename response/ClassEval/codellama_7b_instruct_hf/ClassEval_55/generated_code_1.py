import string

class Manacher:
    def __init__(self, input_string):
        self.input_string = input_string

    def palindromic_length(self, center, diff, string):
        if diff <= 1:
            return 1
        else:
            return self.palindromic_length(center, diff - 1, string)

    def palindromic_string(self):
        longest_substring = ""
        for i in range(len(self.input_string)):
            for j in range(i, len(self.input_string)):
                if self.input_string[i] == self.input_string[j]:
                    substring = self.input_string[i:j+1]
                    if len(substring) > len(longest_substring):
                        longest_substring = substring
        return longest_substring

if __name__ == "__main__":
    manacher = Manacher("ababaxse")
    print(manacher.palindromic_string())