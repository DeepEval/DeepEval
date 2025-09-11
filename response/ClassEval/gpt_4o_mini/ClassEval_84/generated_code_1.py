import json
import re

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
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data

    def read_file(self):
        """
        Read the return the content of self.file_path file.
        :return: the same return as the read() method
        """
        with open(self.file_path, 'r') as f:
            content = f.read()
        return content

    def write_file(self, content):
        """
        Write content into the self.file_path file, and overwrite if the file has already existed.
        :param content: any content
        """
        with open(self.file_path, 'w') as f:
            f.write(content)

    def process_file(self):
        """
        Read the self.file_path file and filter out non-alphabetic characters from the content string.
        Overwrite the after-processed data into the same self.file_path file.
        """
        content = self.read_file()
        # Remove non-alphabetic characters
        processed_content = ''.join(re.findall(r'[a-zA-Z]', content))
        self.write_file(processed_content)
        return processed_content

# Test cases
if __name__ == "__main__":
    # Create a test JSON file
    test_json_content = '{"name": "test", "age": 12}'
    test_file_path = 'test.json'

    # Write test JSON content to the file
    processor = TextFileProcessor(test_file_path)
    processor.write_file(test_json_content)

    # Test read_file_as_json method
    json_data = processor.read_file_as_json()
    print(json_data)  # Expected output: {'name': 'test', 'age': 12}

    # Test read_file method
    raw_content = processor.read_file()
    print(raw_content)  # Expected output: '{"name": "test", "age": 12}'

    # Test write_file method
    processor.write_file('Hello world!')
    print(processor.read_file())  # Expected output: 'Hello world!'

    # Test process_file method
    processor.write_file('Hello 123!')  # Write content to process
    processed_output = processor.process_file()
    print(processed_output)  # Expected output: 'Hello'
    print(processor.read_file())  # Expected output: 'Hello'