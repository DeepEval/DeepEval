import yaml
from utils.match_and_save_code import MatchSaveCode


class Chatlog2Code:
    """
    Extract code from chatlog and save it to the specified folder
    """

    def __init__(self, log_file_path, save_code_path):

        self.log_file_path = log_file_path
        self.save_code_path = save_code_path

    def read_yamldata_to_pycode(self):

        file = open(self.log_file_path, "r", encoding="utf-8")
        file_data = file.read()
        file.close()

        task = yaml.load(file_data, Loader=yaml.FullLoader)
        data = task["response"] 

        data_str = "\n".join(data)

        # extract python code
        match_saver = MatchSaveCode(
            target_folder=self.save_code_path, response=data_str
        )
        matched_code = match_saver.match_code()

        # save code to the specified folder
        task_name = self.log_file_path.split("/")[-2]
        pyfile_name = (
            self.log_file_path.split("/")[-1]
            .replace("yaml", "py")
            .replace("log", "generated_code")
        )
        match_saver.save_code(
            matched_code=matched_code, task_name=task_name, pyfile_name=pyfile_name
        )

        return matched_code


