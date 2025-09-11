import json
import os

class JSONProcessor:
    """
    This is a class to process JSON files, including reading and writing JSON files, 
    as well as processing JSON data by removing a specified key from the JSON object.
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
        except (json.JSONDecodeError, IOError):
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
        except IOError:
            return -1

    def process_json(self, file_path, remove_key):
        """
        Read a JSON file and process the data by removing a specified key and rewrite the modified data back to the file.
        :param file_path: str, the path of the JSON file.
        :param remove_key: str, the key to be removed.
        :return: 1, if the specified key is successfully removed and the data is written back.
                    0, if the file does not exist or the specified key does not exist in the data.
        """
        data = self.read_json(file_path)
        if data == 0 or data == -1 or remove_key not in data:
            return 0
        del data[remove_key]
        return self.write_json(data, file_path)

if __name__ == "__main__":
    json_processor = JSONProcessor()

    # Test read_json
    output = json_processor.read_json('test.json')
    print(f"Read JSON Output: {output}")

    # Test write_json
    sample_data = {'key1': 'value1', 'key2': 'value2'}
    output = json_processor.write_json(sample_data, 'test.json')
    print(f"Write JSON Output: {output}")

    # Test read_json again to verify writing
    output = json_processor.read_json('test.json')
    print(f"Re-Read JSON Output: {output}")

    # Test process_json
    output = json_processor.process_json('test.json', 'key1')
    print(f"Process JSON Output: {output}")

    # Test read_json again to verify processing
    output = json_processor.read_json('test.json')
    print(f"Post-Processing JSON Output: {output}")