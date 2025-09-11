import re

class RegexUtils:
    """
    The class provides to match, find all occurrences, split, and substitute text using regular expressions. It also includes predefined patterns, validating phone numbers and extracting email addresses.
    """

    def match(self, pattern, text):
        """
        Check if the text matches the regular expression
        :param pattern: string, Regular expression pattern
        :param text: string, Text to match
        :return: True or False, representing whether the text matches the regular expression or not
        >>> ru = RegexUtils()
        >>> ru.match(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890")
        True
        """
        try:
            return bool(re.fullmatch(pattern, text))
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return False

    def findall(self, pattern, text):
        """
        Find all matching substrings and return a list of all matching substrings
        :param pattern: string, Regular expression pattern
        :param text: string, Text to match
        :return: list of string, List of all matching substrings
        >>> ru = RegexUtils()
        >>> ru.findall(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767")
        ['123-456-7890', '876-286-9876', '987-762-9767']
        """
        try:
            return re.findall(pattern, text)
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return []

    def split(self, pattern, text):
        """
        Split text based on regular expression patterns and return a list of substrings
        :param pattern: string, Regular expression pattern
        :param text: string, Text to be split
        :return: list of string, List of substrings after splitting
        >>> ru = RegexUtils()
        >>> ru.split(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767")
        ['','abiguygusu ','kjgufwycs ', '']
        """
        try:
            return re.split(pattern, text)
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return []

    def sub(self, pattern, replacement, text):
        """
        Replace the substring matched by a regular expression with the specified string
        :param pattern: string, Regular expression pattern
        :param replacement: Text to replace with
        :param text: string, Text to be replaced
        :return: string, Text after replacement
        >>> ru = RegexUtils()
        >>> ru.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'phone num',  "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767")
        'phone num abiguygusu phone num kjgufwycs phone num'
        """
        try:
            return re.sub(pattern, replacement, text)
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return text

    def generate_email_pattern(self):
        """
        Generate regular expression patterns that match email addresses
        :return: string, regular expression patterns that match email addresses
        >>> ru = RegexUtils()
        >>> ru.generate_email_pattern()
        '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        """
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def generate_phone_number_pattern(self):
        """
        Generate regular expression patterns that match phone numbers
        :return: string, regular expression patterns that match phone numbers
        >>> ru = RegexUtils()
        >>> ru.generate_phone_number_pattern()
        '\b\d{3}-\d{3}-\d{4}\b'
        """
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def generate_split_sentences_pattern(self):
        """
        Generate regular expression patterns that match the middle characters of two sentences
        :return: string, regular expression patterns that match the middle characters of two sentences
        >>> ru = RegexUtils()
        >>> ru.generate_split_sentences_pattern()
        '[.!?][\s]{1,2}(?=[A-Z])'
        """
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_sentences(self, text):
        """
        Split the text into a list of sentences without Punctuation except the last sentence
        :param text: Text to be split
        :return: Split Text List
        >>> ru = RegexUtils()
        >>> ru.split_sentences("Aaa. Bbbb? Ccc!")
        ['Aaa', 'Bbbb', 'Ccc!']
        """
        pattern = self.generate_split_sentences_pattern()
        try:
            return re.split(pattern, text)
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return []

    def validate_phone_number(self, phone_number):
        """
        Verify if the phone number is valid
        :param phone_number: Phone number to be verified
        :return: True or False, indicating whether the phone number is valid
        >>> ru = RegexUtils()
        >>> ru.validate_phone_number("123-456-7890")
        True
        """
        pattern = self.generate_phone_number_pattern()
        try:
            return bool(re.fullmatch(pattern, phone_number))
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return False

    def extract_email(self, text):
        """
        Extract all email addresses from the text
        :param text: string, input text
        :return: list of string, All extracted email addresses
        >>> ru = RegexUtils()
        >>> ru.extract_email("abcdefg@163.com ygusyfysy@126.com wljduyuv@qq.com")
        ['abcdefg@163.com', 'ygusyfysy@126.com', 'wljduyuv@qq.com']
        """
        pattern = self.generate_email_pattern()
        try:
            return re.findall(pattern, text)
        except re.error as e:
            print(f"Invalid pattern: {e}")
            return []

if __name__ == "__main__":
    instance = RegexUtils()

    # Test cases
    print(instance.match(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890"))  # True
    print(instance.findall(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # ['123-456-7890', '876-286-9876', '987-762-9767']
    print(instance.split(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # ['','abiguygusu ','kjgufwycs ', '']
    print(instance.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'phone num', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # 'phone num abiguygusu phone num kjgufwycs phone num'
    print(instance.generate_email_pattern())  # '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    print(instance.generate_phone_number_pattern())  # '\b\d{3}-\d{3}-\d{4}\b'
    print(instance.generate_split_sentences_pattern())  # '[.!?][\s]{1,2}(?=[A-Z])'
    print(instance.split_sentences("Aaa. Bbbb? Ccc!"))  # ['Aaa', 'Bbbb', 'Ccc!']
    print(instance.validate_phone_number("123-456-7890"))  # True
    print(instance.extract_email("abcdefg@163.com ygusyfysy@126.com wljduyuv@qq.com"))  # ['abcdefg@163.com', 'ygusyfysy@126.com', 'wljduyuv@qq.com']