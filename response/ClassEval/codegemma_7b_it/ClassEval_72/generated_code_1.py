import re

class RegexUtils:
    """
    The class provides to match, find all occurrences, split, and substitute text
    using regular expressions. It also includes predefined patterns, validating phone
    numbers and extracting email addresses.
    """

    def match(self, pattern, text):
        """
        Check if the text matches the regular expression
        :param pattern: string, Regular expression pattern
        :param text: string, Text to match
        :return: True or False, representing whether the text matches the regular
        expression or not
        >>> ru = RegexUtils()
        >>> ru.match(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890")
        True
        """
        return re.match(pattern, text) is not None

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
        return re.findall(pattern, text)

    def split(self, pattern, text):
        """
        Split text based on regular expression patterns and return a list of substrings
        :param pattern: string, Regular expression pattern
        :param text: string, Text to be split
        :return: list of string, List of substrings after splitting
        >>> ru = RegexUtils()
        >>> ru.split(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767")
        ['', ' abiguygusu ', ' kjgufwycs ', '']
        """
        return re.split(pattern, text)

    def sub(self, pattern, replacement, text):
        """
        Replace the substring matched by a regular expression with the specified
        string
        :param pattern: string, Regular expression pattern
        :param replacement: Text to replace with
        :param text: string, Text to be replaced
        :return: string, Text after replacement
        >>> ru = RegexUtils()
        >>> ru.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'phone num',  "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767")
        'phone num abiguygusu phone num kjgufwycs phone num'
        """
        return re.sub(pattern, replacement, text)

    def generate_email_pattern(self):
        """
        Generate regular expression patterns that match email addresses
        :return: string, regular expression patterns that match email addresses
        >>> ru = RegexUtils()
        >>> ru.generate_email_pattern()
        '\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        """
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def generate_phone_number_pattern(self):
        """
        Generate regular expression patterns that match phone numbers
        :return: string, regular expression patterns that match phone numbers
        >>> ru = RegexUtils()
        >>> ru.generate_phone_number_pattern()
        '\\b\\d{3}-\\d{3}-\\d{4}\\b'
        """
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def generate_split_sentences_pattern(self):
        """
        Generate regular expression patterns that match the middle characters of
        two sentences
        :return: string, regular expression patterns that match the middle characters
        of two sentences
        >>> ru = RegexUtils()
        >>> ru.generate_split_sentences_pattern()
        '[.!?][\\s]{1,2}(?=[A-Z])'
        """
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_sentences(self, text):
        """
        Split the text into a list of sentences without Punctuation except the
        last sentence
        :param text: Text to be split
        :return: Split Text List
        >>> ru = RegexUtils()
        >>> ru.split_sentences("Aaa. Bbbb? Ccc!")
        ['Aaa', 'Bbbb', 'Ccc!']
        """
        return re.split(self.generate_split_sentences_pattern(), text)

    def validate_phone_number(self, phone_number):
        """
        Verify if the phone number is valid
        :param phone_number: Phone number to be verified
        :return: True or False, indicating whether the phone number is valid
        >>> ru = RegexUtils()
        >>> ru.validate_phone_number("123-456-7890")
        True
        """
        return re.match(self.generate_phone_number_pattern(), phone_number) is not None

    def extract_email(self, text):
        """
        Extract all email addresses from the text
        :param text: string, input text
        :return: list of string, All extracted email addresses
        >>> ru = RegexUtils()
        >>> ru.extract_email("abcdefg@163.com ygusyfysy@126.com wljduyuv@qq.com")
        ['abcdefg@163.com', 'ygusyfysy@126.com', 'wljduyuv@qq.com']
        """
        return re.findall(self.generate_email_pattern(), text)


if __name__ == "__main__":
    ru = RegexUtils()

    # Test case for match() method
    pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    text = "123-456-7890"
    output = ru.match(pattern, text)
    print(output)

    # Test case for findall() method
    pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    text = "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"
    output = ru.findall(pattern, text)
    print(output)

    # Test case for split() method
    pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    text = "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"
    output = ru.split(pattern, text)
    print(output)

    # Test case for sub() method
    pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    replacement = 'phone num'
    text = "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"
    output = ru.sub(pattern, replacement, text)
    print(output)

    # Test case for generate_email_pattern() method
    output = ru.generate_email_pattern()
    print(output)

    # Test case for generate_phone_number_pattern() method
    output = ru.generate_phone_number_pattern()
    print(output)

    # Test case for generate_split_sentences_pattern() method
    output = ru.generate_split_sentences_pattern()
    print(output)

    # Test case for split_sentences() method
    text = "Aaa. Bbbb? Ccc!"
    output = ru.split_sentences(text)
    print(output)

    # Test case for validate_phone_number() method
    phone_number = "123-456-7890"
    output = ru.validate_phone_number(phone_number)
    print(output)

    # Test case for extract_email() method
    text = "abcdefg@163.com ygusyfysy@126.com wljduyuv@qq.com"
    output = ru.extract_email(text)
    print(output)

