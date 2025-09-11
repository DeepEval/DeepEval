import zipfile
import os

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
            return zipfile.ZipFile(self.file_name, 'r')
        except (FileNotFoundError, zipfile.BadZipFile):
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
        except (FileNotFoundError, zipfile.BadZipFile):
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
                zip_ref.extract(file_name, output_path)
            return True
        except (FileNotFoundError, zipfile.BadZipFile, KeyError):
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
                    zip_ref.write(file, os.path.basename(file))
            return True
        except FileNotFoundError:
            return False


# Test Cases
if __name__ == "__main__":
    # Test case for reading a zip file
    zfp = ZipFileProcessor("example.zip")  # Replace with a valid zip file path
    zip_file = zfp.read_zip_file()
    print("Read Zip File:", zip_file is not None)

    # Test case for extracting all files
    result_all = zfp.extract_all("output_directory")  # Replace with a valid output directory
    print("Extract All:", result_all)

    # Test case for extracting a specific file
    result_file = zfp.extract_file("specific_file.txt", "output_directory")  # Replace with a valid file name
    print("Extract Specific File:", result_file)

    # Test case for creating a zip file
    files_to_zip = ["file1.txt", "file2.txt"]  # Replace with valid file paths
    result_create_zip = zfp.create_zip_file(files_to_zip, "output.zip")  # Replace with a valid output zip file
    print("Create Zip File:", result_create_zip)