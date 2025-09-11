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
        with zipfile.ZipFile(self.file_name, "r") as zf:
            return zf.open(self.file_name)

    def extract_all(self, output_path):
        """
        Extract all zip files and place them in the specified path
        :param output_path: string, The location of the extracted file
        :return: True or False, representing whether the extraction operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.extract_all("result/aaa")
        """
        with zipfile.ZipFile(self.file_name, "r") as zf:
            zf.extractall(output_path)
            return True

    def extract_file(self, file_name, output_path):
        """
        Extract the file with the specified name from the zip file and place it in the specified path
        :param file_name:string, The name of the file to be uncompressed
        :param output_path:string, The location of the extracted file
        :return: True or False, representing whether the extraction operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.extract_file("bbb.txt", "result/aaa")
        """
        with zipfile.ZipFile(self.file_name, "r") as zf:
            zf.extract(file_name, output_path)
            return True

    def create_zip_file(self, files, output_file_name):
        """
        Compress the specified file list into a zip file and place it in the specified path
        :param files:list of string, List of files to compress
        :param output_file_name: string, Specified output path
        :return:True or False, representing whether the compression operation was successful
        >>> zfp = ZipFileProcessor("aaa.zip")
        >>> zfp.create_zip_file(["bbb.txt", "ccc,txt", "ddd.txt"], "output/bcd")
        """
        with zipfile.ZipFile(output_file_name, "w") as zf:
            for file in files:
                zf.write(file)
            return True

if __name__ == "__main__":
    zfp = ZipFileProcessor("aaa.zip")
    file = zfp.read_zip_file()
    print(file)
    zfp.extract_all("result/aaa")
    zfp.extract_file("bbb.txt", "result/aaa")
    zfp.create_zip_file(["bbb.txt", "ccc.txt", "ddd.txt"], "output/bcd")