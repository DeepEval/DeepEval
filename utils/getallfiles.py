import os
from natsort import natsorted

class GetAllFiles:
    """
    Get all files in the specified directory
    """
    def __init__(self, directory,file_type):
        self.directory = directory
        self.file_type = file_type

    def get_all_files_in_directory(self,current_directory=None):

        if current_directory is None:
            current_directory = self.directory

        files = os.listdir(current_directory)
        required_files= []

        for file in files:
            full_path = os.path.join(current_directory, file)
            if os.path.isfile(full_path):
                if full_path.endswith(self.file_type):
                    required_files.append(full_path)

            elif os.path.isdir(full_path):
                required_files.extend(self.get_all_files_in_directory(current_directory=full_path))

        sorted_files = natsorted(required_files)

        return sorted_files
