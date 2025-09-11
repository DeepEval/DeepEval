import json

class TextFileProcessor:
    """
    The class handles reading, writing, and processing text files. It can read the file as JSON, read the raw text, write content to the file, and process the file by removing non-alphabetic characters.
    """

    def __init__(self, file_path):
        """
        Initialize the file path.
        :param file_path: str
        """
        self.file_path = file_path

    def read_file_as_json(self):
        """
        Read the self.file_path file as json format.
        if the file content doesn't obey json format, the code will raise error.
        :return data: dict if the file is stored as json format, or str/int/float.. according to the file content otherwise.
        """
        with open(self.file_path, 'r') as file:
            content = file.read()
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                try:
                    return json.loads(f"{{\"content\": {content}}}")['content']
                except Exception as e:
                    return content

    def read_file(self):
        """
        Read the return the content of self.file_path file.
        :return: the same return as the read() method
        """
        with open(self.file_path, 'r') as file:
            return file.read()

    def write_file(self, content):
        """
        Write content into the self.file_path file, and overwrite if the file has already existed.
        :param content: any content
        """
        with open(self.file_path, 'w') as file:
            file.write(content)

    def process_file(self):
        """
        Read the self.file_path file and filter out non-alphabetic characters from the content string.
        Overwrite the after-processed data into the same self.file_path file.
        """
        content = self.read_file()
        processed_content = ''.join(filter(str.isalpha, content))
        self.write_file(processed_content)
        return processed_content

# Test cases
if __name__ == "__main__":
    # Test read_file_as_json
    textFileProcessor = TextFileProcessor('test.json')
    print(textFileProcessor.read_file_as_json())  # Should print {'name': 'test', 'age': 12}

    # Test read_file
    textFileProcessor = TextFileProcessor('test.txt')
    textFileProcessor.write_file('Hello world!')
    print(textFileProcessor.read_file())  # Should print 'Hello world!'

    # Test write_file
    textFileProcessor = TextFileProcessor('test.txt')
    textFileProcessor.write_file('Hello world!')
    print(textFileProcessor.read_file())  # Should print 'Hello world!'

    # Test process_file
    textFileProcessor = TextFileProcessor('test.txt')
    textFileProcessor.write_file('Hello, World! 123')
    print(textFileProcessor.process_file())  # Should print 'HelloWorld'