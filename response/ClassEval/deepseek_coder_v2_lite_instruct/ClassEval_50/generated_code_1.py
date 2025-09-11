import json
import os

class JSONProcessor:
    """
    This is a class to process JSON file, including reading and writing JSON files, as well as processing JSON data by removing a specified key from the JSON object.
    """

    def read_json(self, file_path):
        """
        Read a JSON file and return the data.
        :param file_path: str, the path of the JSON file.
        :return: dict, the data from the JSON file if read successfully, or return -1 if an error occurs during the reading process.
                    return 0 if the file does not exist.
        """
        if not os.path.exists(file_path):
            return 0
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return -1

    def write_json(self, data, file_path):
        """
        Write data to a JSON file and save it to the given path.

        :param data: dict, the data to be written to the JSON file.
        :param file_path: str, the path of the JSON file.
        :return: 1 if the writing process is successful, or -1, if an error occurs during the writing process.
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            return 1
        except Exception as e:
            print(f"Error writing JSON file: {e}")
            return -1

    def process_json(self, file_path, remove_key):
        """
        read a JSON file and process the data by removing a specified key and rewrite the modified data back to the file.

        :param file_path: str, the path of the JSON file.
        :param remove_key: str, the key to be removed.
        :return: 1, if the specified key is successfully removed and the data is written back.
                    0, if the file does not exist or the specified key does not exist in the data.
        """
        data = self.read_json(file_path)
        if data == 0:
            return 0
        elif data == -1:
            return -1
        if remove_key in data:
            del data[remove_key]
            if self.write_json(data, file_path) == 1:
                return 1
            else:
                return -1
        else:
            return 0

# Test cases
if __name__ == "__main__":
    json_processor = JSONProcessor()

    # Test read_json
    print("Testing read_json:")
    print(json_processor.read_json('test.json'))  # Assuming 'test.json' exists with some data
    print(json_processor.read_json('nonexistent.json'))

    # Test write_json
    print("\nTesting write_json:")
    output = json_processor.write_json({'key1': 'value1', 'key2': 'value2'}, 'test_write.json')
    print(output)
    print(json_processor.read_json('test_write.json'))

    # Test process_json
    print("\nTesting process_json:")
    print(json_processor.read_json('test_process.json'))  # Assuming 'test_process.json' exists with some data
    output = json_processor.process_json('test_process.json', 'key1')
    print(output)
    print(json_processor.read_json('test_process.json'))