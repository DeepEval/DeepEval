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
        with open(self.file_path, 'r') as file:
            return json.load(file)

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
        with open(self.file_path, 'r') as file:
            content = file.read()
        processed_content = re.sub('[^a-zA-Z]', '', content)
        with open(self.file_path, 'w') as file:
            file.write(processed_content)
        return processed_content

if __name__ == "__main__":
    # Test case for read_file_as_json
    processor_json = TextFileProcessor('test.json')
    processor_json.write_file('{"name": "test", "age": 12}')  # Preparing the file
    output_json = processor_json.read_file_as_json()
    print("read_file_as_json:", output_json)

    # Test case for read_file
    processor_read = TextFileProcessor('test.txt')
    processor_read.write_file('Hello world!')  # Preparing the file
    output_read = processor_read.read_file()
    print("read_file:", output_read)

    # Test case for write_file
    processor_write = TextFileProcessor('test_write.txt')
    processor_write.write_file('Hello world!')
    output_write = processor_write.read_file()
    print("write_file:", output_write)

    # Test case for process_file
    processor_process = TextFileProcessor('test_process.txt')
    processor_process.write_file('name: test, age: 12')  # Preparing the file
    output_process = processor_process.process_file()
    print("process_file:", output_process)