class HumanEvalPromptDesigner:
    """
    a class to design the prompt for the humaneval task.
    """
    def __init__(self):
        pass

    def prompt(self, task_requirement):
        """
        Prompt: prefix-<requirement>-->code
        """
        backgroud = """
    As a developer specializing in software engineering, you are expected to complete the code 
that meets the requirement of the following task:\n"""

        new_task_requirement = "    " +task_requirement

        code_format = """
    Please import all necessary packages, then complete python code in the 'method()' function and return 
the final 'output' if needed. Additionally, please provide a test case for validation.
    ```python
    def method():

        return output
    ```
    """
        prompt = backgroud + new_task_requirement + code_format

        return prompt

