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
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError:
            with open(self.file_path, 'r') as file:
                data = file.read()
            return data

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
        processed_content = ''.join(char for char in content if char.isalpha() or char.isspace())
        self.write_file(processed_content)

# Test cases

if __name__ == "__main__":
    textFileProcessor = TextFileProcessor('test.json')

    # Test case 1: Read file as JSON
    json_data = textFileProcessor.read_file_as_json()
    print(json_data)

    # Test case 2: Read file as text
    text_data = textFileProcessor.read_file()
    print(text_data)

    # Test case 3: Write to file
    textFileProcessor.write_file('Hello world!')

    # Test case 4: Process file
    textFileProcessor.process_file()
    processed_data = textFileProcessor.read_file()
    print(processed_data)