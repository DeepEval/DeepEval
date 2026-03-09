class CompilelogBasedRepairPrompt:
    def __init__(self):
        pass
    def prompt(self, buggy_code, messages: str) -> str:
        background = """
You are a Python code runtime error repair expert. Your task is to help fix the provided Python code based on the compile log.
"""
        instruction = """
Check the above origin code and compile log, fix the errors, and output the corrected code.

Note: Do not introduce any new layers; no additional layers.X(...) or tf.keras.layers.X(...) statements beyond those already in the original code.
"""

        prompt = background + f"\n\nThe Origin Code:\n```python\n{buggy_code}```\n\nCompile Log:\n{messages}" + f"\n{instruction}\n " 

        return prompt