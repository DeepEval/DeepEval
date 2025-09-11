import json
import os

class JSONProcessor:
    def read_json(self, file_path):
        if not os.path.isfile(file_path):
            return -1

        with open(file_path, 'r') as file:
            data = json.load(file)
            return data

    def write_json(self, data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file)
        return 1

    def process_json(self, file_path, remove_key):
        if not os.path.isfile(file_path):
            return 0

        data = self.read_json(file_path)
        if remove_key in data:
            del data[remove_key]
            self.write_json(data, file_path)
            return 1
        else:
            return 0

if __name__ == "__main__":
    json_processor = JSONProcessor()

    # Test case 1: Reading a JSON file
    data = json_processor.read_json('test.json')
    print(data)

    # Test case 2: Writing data to a JSON file
    data = {'key1': 'value1', 'key2': 'value2'}
    json_processor.write_json(data, 'test.json')
    print(json_processor.read_json('test.json'))

    # Test case 3: Processing a JSON file
    data = json_processor.read_json('test.json')
    json_processor.process_json('test.json', 'key1')
    print(json_processor.read_json('test.json'))