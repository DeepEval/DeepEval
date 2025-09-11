from utils.getallfiles import GetAllFiles
from utils.extract_humaneval_text import extract_text
from utils.read_json_benchmark import read_classeval


class ClassEvalPromptDesigner:
    """
    a class to design the prompt for the classeval task.
    """
    def __init__(self):
        pass

    def prompt(self, task_requirement):
        """
        Prompt：prefix-<requirement>-->code
        """
        backgroud_role = """    
    As a developer specializing in software engineering, you are expected to complete the code 
that meets the requirement of the class:\n\n"""

        requirement = task_requirement

        hint = """
    Please import all necessary packages, then complete python code in the class and return 
the 'output' for each method if needed. Additionally, please provide a test case for each method
to validate its code executability.
    ```python
    class ClassName:

        def method_name(self, args):
            # Your code here
            pass
    
    if __name__ == "__main__":
        instance = ClassName()
        # Test case
        output = instance.method_name(args)
        print(output)
    ```
    """
        prompt = backgroud_role + requirement + hint

        return prompt