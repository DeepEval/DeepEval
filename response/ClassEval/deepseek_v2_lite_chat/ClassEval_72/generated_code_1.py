import re

class RegexUtils:
    def match(self, pattern, text):
        """
        Check if the text matches the regular expression
        :param pattern: string, Regular expression pattern
        :param text: string, Text to match
        :return: True or False, representing whether the text matches the regular expression or not
        """
        return bool(re.match(pattern, text))

    def findall(self, pattern, text):
        """
        Find all matching substrings and return a list of all matching substrings
        :param pattern: string, Regular expression pattern
        :param text: string, Text to match
        :return: list of string, List of all matching substrings
        """
        return re.findall(pattern, text)

    def split(self, pattern, text):
        """
        Split text based on regular expression patterns and return a list of substrings
        :param pattern: string, Regular expression pattern
        :param text: string, Text to be split
        :return: list of string, List of substrings after splitting
        """
        return re.split(pattern, text)

    def sub(self, pattern, replacement, text):
        """
        Replace the substring matched by a regular expression with the specified string
        :param pattern: string, Regular expression pattern
        :param replacement: Text to replace with
        :param text: string, Text to be replaced
        :return: string, Text after replacement
        """
        return re.sub(pattern, replacement, text)

    def generate_email_pattern(self):
        """
        Generate regular expression patterns that match email addresses
        :return: string, regular expression patterns that match email addresses
        """
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def generate_phone_number_pattern(self):
        """
        Generate regular expression patterns that match phone numbers
        :return: string, regular expression patterns that match phone numbers
        """
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def generate_split_sentences_pattern(self):
        """
        Generate regular expression patterns that match the middle characters of two sentences
        :return: string, regular expression patterns that match the middle characters of two sentences
        """
        return '[.!?][\s]{1,2}(?=[A-Z])'

    def split_sentences(self, text):
        """
        Split the text into a list of sentences without Punctuation except the last sentence
        :param text: Text to be split
        :return: Split Text List
        """
        return text.split('[.!?]\s')[:-1]

    def validate_phone_number(self, phone_number):
        """
        Verify if the phone number is valid
        :param phone_number: Phone number to be verified
        :return: True or False, indicating whether the phone number is valid
        """
        return bool(re.match(self.generate_phone_number_pattern(), phone_number))

    def extract_email(self, text):
        """
        Extract all email addresses from the text
        :param text: string, input text
        :return: list of string, All extracted email addresses
        """
        return re.findall(self.generate_email_pattern(), text)

# Test cases
if __name__ == "__main__":
    ru = RegexUtils()
    print(ru.match(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890"))  # True
    print(ru.findall(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # ['123-456-7890', '876-286-9876', '987-762-9767']
    print(ru.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'phone num', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # 'phone num abiguygusu phone num kjgufwycs phone num'
    print(ru.generate_email_pattern())  # '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    print(ru.generate_phone_number_pattern())  # '\b\d{3}-\d{3}-\d{4}\b'
    print(ru.generate_split_sentences_pattern())  # '[.!?][\s]{1,2}(?=[A-Z])'
    print(ru.split_sentences("Aaa. Bbbb? Ccc!"))  # ['Aaa', 'Bbbb', 'Ccc!']
    print(ru.validate_phone_number("123-456-7890"))  # True
    print(ru.extract_email("abcdefg@163.com ygusyfysy@126.com wljduyuv@qq.com"))  # ['abcdefg@163.com', 'ygusyfysy@126.com', 'wljduyuv@qq.com']