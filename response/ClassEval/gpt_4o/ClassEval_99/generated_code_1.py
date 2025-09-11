import zipfile
import os

class ZipFileProcessor:
    """
    This is a compressed file processing class that provides the ability to read and decompress compressed files
    """

    def __init__(self, file_name):
        """
        Initialize file name
        :param file_name:string
        """
        self.file_name = file_name

    def read_zip_file(self):
        """
        Get open file object
        :return:If successful, returns the open file object; otherwise, returns None
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> file = zfp.read_zip_file()
        """
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                return zip_ref
        except FileNotFoundError:
            print(f"File {self.file_name} not found.")
            return None
        except zipfile.BadZipFile:
            print(f"File {self.file_name} is not a zip file.")
            return None

    def extract_all(self, output_path):
        """
        Extract all zip files and place them in the specified path
        :param output_path: string, The location of the extracted file
        :return: True or False, representing whether the extraction operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.extract_all("result/aaa")
        """
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def extract_file(self, file_name, output_path):
        """
        Extract the file with the specified name from the zip file and place it in the specified path
        :param file_name:string, The name of the file to be uncompressed
        :param output_path:string, The location of the extracted file
        :return: True or False, representing whether the extraction operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.extract_file("bbb.txt", "result/aaa")
        """
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                zip_ref.extract(file_name, output_path)
            return True
        except KeyError:
            print(f"File {file_name} not found in the zip.")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def create_zip_file(self, files, output_file_name):
        """
        Compress the specified file list into a zip file and place it in the specified path
        :param files:list of string, List of files to compress
        :param output_file_name: string, Specified output path
        :return:True or False, representing whether the compression operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.create_zip_file(["bbb.txt", "ccc.txt", "ddd.txt"], "output/bcd.zip")
        """
        try:
            with zipfile.ZipFile(output_file_name, 'w') as zip_ref:
                for file in files:
                    zip_ref.write(file, os.path.basename(file))
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False


if __name__ == "__main__":
    # Test case for read_zip_file
    zfp = ZipFileProcessor("test.zip")
    zip_file = zfp.read_zip_file()
    print(f"Read zip file: {zip_file is not None}")

    # Test case for extract_all
    result = zfp.extract_all("output_path")
    print(f"Extract all: {result}")

    # Test case for extract_file
    result = zfp.extract_file("test.txt", "output_path")
    print(f"Extract file: {result}")

    # Test case for create_zip_file
    result = zfp.create_zip_file(["test1.txt", "test2.txt"], "new_test.zip")
    print(f"Create zip file: {result}")