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
        except Exception as e:
            print(f"Error opening zip file: {e}")
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
            print(f"Error extracting zip file: {e}")
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
        except Exception as e:
            print(f"Error extracting file from zip: {e}")
            return False

    def create_zip_file(self, files, output_file_name):
        """
        Compress the specified file list into a zip file and place it in the specified path
        :param files: list of string, List of files to compress
        :param output_file_name: string, Specified output path
        :return: True or False, representing whether the compression operation was successful
        """
        try:
            with zipfile.ZipFile(output_file_name, 'w') as zip_file:
                for file in files:
                    zip_file.write(file)
            return True
        except Exception as e:
            print(f"Error creating zip file: {e}")
            return False

# Test cases
if __name__ == "__main__":
    # Test read_zip_file
    zfp = ZipFileProcessor("test.zip")
    file = zfp.read_zip_file()
    print(f"File object: {file}")

    # Test extract_all
    success = zfp.extract_all("extracted_files")
    print(f"Extract all successful: {success}")

    # Test extract_file
    success = zfp.extract_file("file_to_extract.txt", "extracted_files")
    print(f"Extract file successful: {success}")

    # Test create_zip_file
    success = zfp.create_zip_file(["file_to_zip.txt", "another_file.txt"], "output.zip")
    print(f"Create zip file successful: {success}")