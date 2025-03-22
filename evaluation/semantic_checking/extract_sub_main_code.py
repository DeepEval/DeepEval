import ast
import astor


class ExtractMainSubCodes:

    def __init__(self):
        pass

    def preprocess_sub_main_code(self, code):
        """
        Preprocess the code, extract the main function code and subfunction code
        """
        code = ast.parse(code)
        subfunctions_code = []
        for node in code.body:
            if isinstance(node, ast.FunctionDef) and node.name == "dl_model":

                main_code = astor.to_source(node)
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):  
                        function_code = astor.to_source(sub_node)
                        subfunctions_code.append(function_code.strip())

        return main_code, subfunctions_code

