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
        :return: dict, the data from the JSON file if read successfully, or return -1 if an error occurs during the reading process.
                    return 0 if the file does not exist.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            return 0
        except json.JSONDecodeError as e:
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
                json.dump(data, file)
                return 1
        except Exception as e:
            print(f"Error writing JSON file: {e}")
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
        if data == 0:
            return 0
        if remove_key in data:
            del data[remove_key]
            return self.write_json(data, file_path)
        else:
            return 0


if __name__ == "__main__":
    json_processor = JSONProcessor()

    # Test case for read_json method
    print("Test case for read_json method:")
    file_path = 'test.json'
    data = json_processor.read_json(file_path)
    if isinstance(data, dict):
        print(f"Data from JSON file: {data}")
    elif data == 0:
        print(f"File {file_path} does not exist.")
    else:
        print(f"Error reading JSON file: {data}")

    # Create a test.json file
    data = {'name': 'test', 'age': 14}
    json_processor.write_json(data, file_path)

    # Test case for write_json method
    print("\nTest case for write_json method:")
    file_path = 'test2.json'
    data = {'key1': 'value1', 'key2': 'value2'}
    result = json_processor.write_json(data, file_path)
    if result == 1:
        print(f"Data written to JSON file: {data}")
    else:
        print(f"Error writing JSON file: {result}")

    # Test case for process_json method
    print("\nTest case for process_json method:")
    file_path = 'test.json'
    remove_key = 'name'
    result = json_processor.process_json(file_path, remove_key)
    if result == 1:
        print(f"Key '{remove_key}' removed from JSON file.")
    elif result == 0:
        print(f"Key '{remove_key}' does not exist in JSON file.")
    else:
        print(f"Error processing JSON file: {result}")

    # Remove the test.json and test2.json files
    os.remove('test.json')
    os.remove('test2.json')