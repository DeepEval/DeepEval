import json
import os

class JSONProcessor:
    """
    This is a class to process JSON file, including reading and writing JSON files,
    as well as processing JSON data by removing a specified key from the JSON object.
    """

    def read_json(self, file_path):
        """
        Read a JSON file and return the data.
        :param file_path: str, the path of the JSON file.
        :return: dict, the data from the JSON file if read successfully, 
                or return -1 if an error occurs during the reading process, 
                return 0 if the file does not exist.
        """
        if not os.path.exists(file_path):
            return 0

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            return -1

    def write_json(self, data, file_path):
        """
        Write data to a JSON file and save it to the given path.
        :param data: dict, the data to be written to the JSON file.
        :param file_path: str, the path of the JSON file.
        :return: 1 if the writing process is successful, 
                or -1, if an error occurs during the writing process.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            return 1
        except Exception as e:
            return -1

    def process_json(self, file_path, remove_key):
        """
        read a JSON file and process the data by removing a specified key 
        and rewrite the modified data back to the file.
        :param file_path: str, the path of the JSON file.
        :param remove_key: str, the key to be removed.
        :return: 1, if the specified key is successfully removed 
                and the data is written back.
                0, if the file does not exist or the specified key does not exist in the data.
        """
        if not os.path.exists(file_path):
            return 0

        data = self.read_json(file_path)
        if data == -1 or remove_key not in data:
            return 0

        data.pop(remove_key)
        return self.write_json(data, file_path)

# Test cases
if __name__ == "__main__":
    processor = JSONProcessor()
    
    # Test write_json
    output = processor.write_json({'name': 'test', 'age': 14}, 'test.json')
    print(f'write_json output: {output}')  # Expected: 1

    # Test read_json
    output = processor.read_json('test.json')
    print(f'read_json output: {output}')  # Expected: {'name': 'test', 'age': 14}

    # Test process_json (removing key)
    output = processor.process_json('test.json', 'name')
    print(f'process_json output: {output}')  # Expected: 1
    output = processor.read_json('test.json')
    print(f'read_json after process_json output: {output}')  # Expected: {'age': 14}

    # Test process_json (removing a non-existing key)
    output = processor.process_json('test.json', 'non_existing_key')
    print(f'process_json (non-existing key) output: {output}')  # Expected: 0