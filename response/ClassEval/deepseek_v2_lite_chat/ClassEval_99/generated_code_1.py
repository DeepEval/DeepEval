import zipfile

class ZipFileProcessor:
    """
    This is a compressed file processing class that provides the ability to read and decompress compressed files
    """

    def __init__(self, file_name):
        """
        Initialize file name
        :param file_name: string
        """
        self.file_name = file_name

    def read_zip_file(self):
        """
        Get open file object
        :return: If successful, returns the open file object; otherwise, returns None
        """
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                content = zip_ref.read('test.txt')  # Assuming 'test.txt' is a test file in the zip
                return content
        except zipfile.BadZipFile as e:
            print(f"Error: {e}")
            return None

    def extract_all(self, output_path):
        """
        Extract all zip files and place them in the specified path
        :param output_path: string, The location of the extracted file
        :return: True or False, representing whether the extraction operation was successful
        """
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def extract_file(self, file_name, output_path):
        """
        Extract the file with the specified name from the zip file and place it in the specified path
        :param file_name: string, The name of the file to be uncompressed
        :param output_path: string, The location of the extracted file
        :return: True or False, representing whether the extraction operation was successful
        """
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                zip_ref.extract(file_name, path=output_path)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def create_zip_file(self, files, output_file_name):
        """
        Compress the specified file list into a zip file and place it in the specified path
        :param files: list of string, List of files to compress
        :param output_file_name: string, Specified output path
        :return: True or False, representing whether the compression operation was successful
        """
        try:
            with zipfile.ZipFile(output_file_name, 'w') as zip_ref:
                for file in files:
                    zip_ref.write(file)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

# Test cases
if __name__ == "__main__":
    zfp = ZipFileProcessor("example.zip")
    print(zfp.read_zip_file())  # Output: content of 'test.txt' if present
    print(zfp.extract_all("test_extraction"))  # Output: True if successful
    print(zfp.extract_file("test.txt", "test_extraction/subdir"))  # Output: True if successful
    print(zfp.create_zip_file(["test1.txt", "test2.txt"], "new_example.zip"))  # Output: True if successful