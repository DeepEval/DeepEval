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
        >>> json.read_json('test.json')
        {'name': 'test', 'age': 14}
        """
        if not os.path.exists(file_path):
            return 0
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            return -1

    def write_json(self, data, file_path):
        """
        Write data to a JSON file and save it to the given path.

        :param data: dict, the data to be written to the JSON file.
        :param file_path: str, the path of the JSON file.
        :return: 1 if the writing process is successful, or -1, if an error occurs during the writing process.
        >>> json.write_json({'key1': 'value1', 'key2': 'value2'}, 'test.json')
        1
        >>> json.read_json('test.json')
        {'key1': 'value1', 'key2': 'value2'}
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            return 1
        except Exception as e:
            print(f"Error writing to file: {e}")
            return -1

    def process_json(self, file_path, remove_key):
        """
        read a JSON file and process the data by removing a specified key and rewrite the modified data back to the file.

        :param file_path: str, the path of the JSON file.
        :param remove_key: str, the key to be removed.
        :return: 1, if the specified key is successfully removed and the data is written back.
                    0, if the file does not exist or the specified key does not exist in the data.
        >>> json.read_json('test.json')
        {'key1': 'value1', 'key2': 'value2'}
        >>> json.process_json('test.json', 'key1')
        1
        >>> json.read_json('test.json')
        {'key2': 'value2'}
        """
        data = self.read_json(file_path)
        if data == 0:
            return 0
        if remove_key in data:
            del data[remove_key]
            self.write_json(data, file_path)
            return 1
        else:
            return 0

if __name__ == "__main__":
    instance = JSONProcessor()
    # Test case for read_json
    output = instance.read_json('test.json')
    print(f"Output of read_json: {output}")

    # Test case for write_json
    instance.write_json({'key1': 'value1', 'key2': 'value2'}, 'test.json')
    output = instance.read_json('test.json')
    print(f"Output of write_json and read_json: {output}")

    # Test case for process_json
    output = instance.process_json('test.json', 'key1')
    print(f"Output of process_json: {output}")
    output = instance.read_json('test.json')
    print(f"Output of read_json after process_json: {output}")