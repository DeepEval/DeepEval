import zipfile


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
            print(f"Error: File '{self.file_name}' not found.")
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
            print(f"Error extracting files: {e}")
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
        except Exception as e:
            print(f"Error extracting file '{file_name}': {e}")
            return False

    def create_zip_file(self, files, output_file_name):
        """
        Compress the specified file list into a zip file and place it in the specified path
        :param files:list of string, List of files to compress
        :param output_file_name: string, Specified output path
        :return:True or False, representing whether the compression operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.create_zip_file(["bbb.txt", "ccc,txt", "ddd.txt"], "output/bcd")
        """
        try:
            with zipfile.ZipFile(output_file_name, 'w') as zip_ref:
                for file in files:
                    zip_ref.write(file)
            return True
        except Exception as e:
            print(f"Error creating zip file: {e}")
            return False


if __name__ == "__main__":
    # Test cases
    zfp = ZipFileProcessor("test.zip")  

    # read_zip_file
    file = zfp.read_zip_file()
    print(f"File object: {file}")

    # extract_all
    zfp.extract_all("extracted_files")
    print(f"Extraction result: {zfp.extract_all('extracted_files')}")

    # extract_file
    zfp.extract_file("test.txt", "extracted_files/test")
    print(f"Extraction result: {zfp.extract_file('test.txt', 'extracted_files/test')}")

    # create_zip_file
    zfp.create_zip_file(["test.txt", "another_file.txt"], "new_zip.zip")
    print(f"Zip creation result: {zfp.create_zip_file(['test.txt', 'another_file.txt'], 'new_zip.zip')}")