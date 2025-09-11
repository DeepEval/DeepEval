import json
import os

class JSONProcessor:
    """
    This is a class to process JSON files, including reading and writing JSON files, as well as processing JSON data by removing a specified key from the JSON object.
    """

    def read_json(self, file_path):
        """
        Read a JSON file and return the data.
        :param file_path: str, the path of the JSON file.
        :return: dict or -1, depending on success or failure.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            return 0
        except json.JSONDecodeError:
            return -1

    def write_json(self, data, file_path):
        """
        Write data to a JSON file and save it to the given path.
        :param data: dict, the data to be written to the JSON file.
        :param file_path: str, the path of the JSON file.
        :return: 1 if successful, or -1 if an error occurs.
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return 1
        except IOError:
            return -1

    def process_json(self, file_path, remove_key):
        """
        Read a JSON file and process the data by removing a specified key and rewrite the modified data back to the file.
        :param file_path: str, the path of the JSON file.
        :param remove_key: str, the key to be removed.
        :return: 1 if successful, or 0 if the file does not exist or the key is not found.
        """
        if not os.path.exists(file_path):
            return 0

        data = self.read_json(file_path)
        if data is 0:
            return 0
        if data is -1:
            return 0

        if remove_key not in data:
            return 0

        del data[remove_key]
        self.write_json(data, file_path)
        return 1

# Test cases
if __name__ == "__main__":
    json_processor = JSONProcessor()
    test_data = {'name': 'test', 'age': 14}

    # Test read_json
    print(json_processor.read_json('test.json'))  # Should return the initial data

    # Test write_json
    print(json_processor.write_json(test_data, 'test.json'))  # Should return 1
    print(json_processor.read_json('test.json'))  # Should return the modified data

    # Test process_json
    print(json_processor.process_json('test.json', 'age'))  # Should return 1
    print(json_processor.read_json('test.json'))  # Should return {'name': 'test'}