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
            return zipfile.ZipFile(self.file_name)
        except FileNotFoundError:
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
            zip_ref = zipfile.ZipFile(self.file_name, 'r')
            zip_ref.extractall(output_path)
            zip_ref.close()
            return True
        except Exception as e:
            print(e)
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
            zip_ref = zipfile.ZipFile(self.file_name, 'r')
            zip_ref.extract(file_name, output_path)
            zip_ref.close()
            return True
        except Exception as e:
            print(e)
            return False

    def create_zip_file(self, files, output_file_name):
        """
        Compress the specified file list into a zip file and place it in the specified path
        :param files:list of string, List of files to compress
        :param output_file_name: string, Specified output path
        :return:True or False, representing whether the compression operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.create_zip_file(["bbb.txt", "ccc.txt", "ddd.txt"], "output/bcd")
        """
        try:
            zip_ref = zipfile.ZipFile(output_file_name, 'w')
            for file in files:
                zip_ref.write(file)
            zip_ref.close()
            return True
        except Exception as e:
            print(e)
            return False

if __name__ == "__main__":
    # Test case for read_zip_file method
    zfp = ZipFileProcessor("aaa.zip")
    file = zfp.read_zip_file()
    print(file)

    # Test case for extract_all method
    zfp = ZipFileProcessor("aaa.zip")
    output_path = "result/aaa"
    if zfp.extract_all(output_path):
        print(f"Files extracted successfully to {output_path}")
    else:
        print("Failed to extract files")

    # Test case for extract_file method
    zfp = ZipFileProcessor("aaa.zip")
    file_name = "bbb.txt"
    output_path = "result/aaa"
    if zfp.extract_file(file_name, output_path):
        print(f"File {file_name} extracted successfully to {output_path}")
    else:
        print(f"Failed to extract file {file_name}")

    # Test case for create_zip_file method
    zfp = ZipFileProcessor(None)
    files = ["bbb.txt", "ccc.txt", "ddd.txt"]
    output_file_name = "output/bcd.zip"
    if zfp.create_zip_file(files, output_file_name):
        print(f"Zip file created successfully at {output_file_name}")
    else:
        print(f"Failed to create zip file at {output_file_name}")