import re

class RegexUtils:
    """
    The class provides to match, find all occurrences, split, and substitute text using regular expressions. It also includes predefined patterns, validating phone numbers and extracting email addresses.
    """

    def match(self, pattern, text):
        """
        Check if the text matches the regular expression.
        """
        return bool(re.match(pattern, text))

    def findall(self, pattern, text):
        """
        Find all matching substrings and return a list of all matching substrings.
        """
        return re.findall(pattern, text)

    def split(self, pattern, text):
        """
        Split text based on regular expression patterns and return a list of substrings.
        """
        return re.split(pattern, text)

    def sub(self, pattern, replacement, text):
        """
        Replace the substring matched by a regular expression with the specified string.
        """
        return re.sub(pattern, replacement, text)

    def generate_email_pattern(self):
        """
        Generate regular expression patterns that match email addresses.
        """
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def generate_phone_number_pattern(self):
        """
        Generate regular expression patterns that match phone numbers.
        """
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def generate_split_sentences_pattern(self):
        """
        Generate regular expression patterns that match the middle characters of two sentences.
        """
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_sentences(self, text):
        """
        Split the text into a list of sentences without punctuation except the last sentence.
        """
        pattern = self.generate_split_sentences_pattern()
        return re.split(pattern, text)

    def validate_phone_number(self, phone_number):
        """
        Verify if the phone number is valid.
        """
        pattern = self.generate_phone_number_pattern()
        return bool(re.match(pattern, phone_number))

    def extract_email(self, text):
        """
        Extract all email addresses from the text.
        """
        pattern = self.generate_email_pattern()
        return re.findall(pattern, text)

if __name__ == "__main__":
    ru = RegexUtils()
    
    # Test case for match
    print(ru.match(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890"))  # Output: True
    
    # Test case for findall
    print(ru.findall(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # Output: ['123-456-7890', '876-286-9876', '987-762-9767']
    
    # Test case for split
    print(ru.split(r'\b\d{3}-\d{3}-\d{4}\b', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # Output: ['', ' abiguygusu ', ' kjgufwycs ', '']
    
    # Test case for sub
    print(ru.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'phone num', "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"))  # Output: 'phone num abiguygusu phone num kjgufwycs phone num'
    
    # Test case for generate_email_pattern
    print(ru.generate_email_pattern())  # Output: '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Test case for generate_phone_number_pattern
    print(ru.generate_phone_number_pattern())  # Output: '\b\d{3}-\d{3}-\d{4}\b'
    
    # Test case for generate_split_sentences_pattern
    print(ru.generate_split_sentences_pattern())  # Output: '[.!?][\s]{1,2}(?=[A-Z])'
    
    # Test case for split_sentences
    print(ru.split_sentences("Aaa. Bbbb? Ccc!"))  # Output: ['Aaa', 'Bbbb', 'Ccc!']
    
    # Test case for validate_phone_number
    print(ru.validate_phone_number("123-456-7890"))  # Output: True
    
    # Test case for extract_email
    print(ru.extract_email("abcdefg@163.com ygusyfysy@126.com wljduyuv@qq.com"))  # Output: ['abcdefg@163.com', 'ygusyfysy@126.com', 'wljduyuv@qq.com']