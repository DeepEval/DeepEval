class ZeroshotPromptDesigner:
    """
    ZeroshotPromptDesigner: a class to design the zero-shot prompting.
    Prompt: prefix-<requirement>-->code
    """

    def __init__(self):
        pass

    def prompt(self, task_requirement):

        backgroud = """
    As a developer specializing in deep learning, you are expected to complete the code 
using Functional APIs of Keras, ensuring it meets the requirement of the following task: \n"""

        new_task_requirement = "    " + f'"{task_requirement}"'

        code_format = """ 
    Please import Keras and all necessary packages, then complete python code in the 'dl_model()' 
function and return the constructed 'model'.
    ```python
    def dl_model():

        return model
    ```
    """

        prompt = backgroud + new_task_requirement + code_format

        return prompt


