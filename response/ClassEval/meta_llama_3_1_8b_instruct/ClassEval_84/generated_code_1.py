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
                return file.read()

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
        processed_content = ''.join(char for char in content if char.isalpha())
        self.write_file(processed_content)


if __name__ == "__main__":
    text_file_processor = TextFileProcessor('test.json')
    
    # Test case for read_file_as_json
    data = text_file_processor.read_file_as_json()
    print("Test case for read_file_as_json:")
    print("Output:", data)
    print("Type of output:", type(data))
    
    # Test case for read_file
    content = text_file_processor.read_file()
    print("\nTest case for read_file:")
    print("Output:", content)
    print("Type of output:", type(content))
    
    # Test case for write_file
    text_file_processor.write_file('Hello world!')
    content = text_file_processor.read_file()
    print("\nTest case for write_file:")
    print("Output:", content)
    print("Type of output:", type(content))
    
    # Test case for process_file
    text_file_processor.process_file()
    content = text_file_processor.read_file()
    print("\nTest case for process_file:")
    print("Output:", content)
    print("Type of output:", type(content))