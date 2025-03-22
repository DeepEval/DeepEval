class MLEvalPromptDesigner:
    """
    a class to design the prompt for the MLEval task.
    """
    def __init__(self):
        pass

    def prompt(self, task_requirement):
        """
        Prompt: prefix-<requirement>-->code
        """
        backgroud = """
    As a developer specializing in machine learning, you are expected to complete the code that meets
the requirement of the following task:\n"""

        task_new_requirement = "    "+f'"{task_requirement.strip()}"'

        code_format = """ 
    Please import all necessary packages, then complete python code in the 'method()' function and return 
the final 'output' if needed. Additionally, please call the generated 'method()' for validation.
    ```python
    def method():

        return output
    ```
    """

        prompt = backgroud + task_new_requirement + code_format

        return prompt

