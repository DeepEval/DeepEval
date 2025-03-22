from utils.getallfiles import GetAllFiles


class GetModelsFiles2Module(GetAllFiles):
    """
    Class for obtaining DeepEval model files: Batch read the full path of python files and convert them to module names.
    """

    def __init__(self, directory, file_type, project_name):
        super().__init__(directory, file_type)
        self.project_name = project_name

    def get_processing_models_files(self):
        get_all_files = GetAllFiles(self.directory, self.file_type)
        model_files_names = get_all_files.get_all_files_in_directory()
        model_module_modified_files = []

        for model_file in model_files_names:
            index = model_file.find(self.project_name)
            if index != -1:
                extracted_string = model_file[
                    index + len(self.project_name) + 1 : -3
                ]
                model_module_modified_file = extracted_string.replace(
                    "/", "."
                )
                model_module_modified_files.append(model_module_modified_file)
            else:
                print("String 'DeepEval' not found in the path.")

        return model_module_modified_files

