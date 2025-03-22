from utils.getallfiles import GetAllFiles


class ReadPythonCode(GetAllFiles):
    """
    Read python code in batch.
    """
    def __init__(self, directory, file_type):
        super().__init__(directory, file_type)

    def get_pyfiles(self):
        get_all_files = GetAllFiles(self.directory, '.py')
        pyfiles = get_all_files.get_all_files_in_directory()

        return pyfiles

    def read_source_code(self,pyfile):
        with open(pyfile,"r") as file:
            python_code=file.read()
        return python_code
        