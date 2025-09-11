import json

class TextFileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file_as_json(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f'{self.file_path} is not a JSON file')
        return data

    def read_file(self):
        with open(self.file_path, 'r') as f:
            return f.read()

    def write_file(self, content):
        with open(self.file_path, 'w') as f:
            f.write(content)

    def process_file(self):
        with open(self.file_path, 'r') as f:
            content = f.read()
        processed_content = self._remove_non_alphabetic_chars(content)
        with open(self.file_path, 'w') as f:
            f.write(processed_content)

    def _remove_non_alphabetic_chars(self, content):
        return ''.join(c for c in content if c.isalpha())

if __name__ == '__main__':
    text_file_processor = TextFileProcessor('test.json')
    print(text_file_processor.read_file_as_json())
    print(text_file_processor.read_file())
    text_file_processor.write_file('Hello world!')
    print(text_file_processor.read_file())
    text_file_processor.process_file()
    print(text_file_processor.read_file())