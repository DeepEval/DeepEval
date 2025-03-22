import ast
import os
import jsonpickle

class Extract_Code_Info:
    """
    Extract the key information in the code, including the function name, parameters, return value, etc., form a dictionary format data, and save it as a json file
    """
    def __init__(self):
        super().__init__()

    def get_argument_value(self, arg):
        """
        Extract specific parameter values based on different parameter data types, including strings, numbers, lists, tuples, dictionaries, attributes, function calls, etc.
        """
        if isinstance(arg, ast.NameConstant):
            return arg.value
        elif isinstance(arg, ast.Str):
            return arg.s
        elif isinstance(arg, ast.Num):
            return arg.n
        elif isinstance(arg, ast.UnaryOp): 
            return self.get_argument_value(arg.operand)
        elif isinstance(arg, ast.List):
            return [self.get_argument_value(item) for item in arg.elts]
        elif isinstance(arg, ast.Tuple):
            return tuple([self.get_argument_value(item) for item in arg.elts])
        elif isinstance(arg, ast.Dict):
            return {self.get_argument_value(k): self.get_argument_value(v) for k, v in zip(arg.keys, arg.values)}
        elif isinstance(arg, ast.Attribute):
            return arg.attr
        elif isinstance(arg, ast.Call): 
            return self.get_argument_value(arg)
        elif isinstance(arg, ast.Name):
            return arg.id
        else:
            return None

    def extract_assign_variable_name(self, node):
        variable_name = None
        for target in node.targets:
            if isinstance(target, ast.Name):
                variable_name = target.id
        return variable_name

    def extract_func_name(self, node):

        def extract_attribute_name(node):
            if isinstance(node, ast.Attribute):
                return node.attr 

            elif isinstance(node, ast.Name):
                return node.id
            else:
                return None

        if isinstance(node, ast.Call):
            return self.extract_func_name(node.func)
        elif isinstance(node, ast.Attribute):
            return extract_attribute_name(node)
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None

    def extract_function_internal_params(self, call_node):
        function_internal_params = {}
        external_arg = self.extract_function_external_params(call_node)

        if external_arg == 'None':
            for internal_arg in call_node.keywords:
                function_internal_params[internal_arg.arg] = self.get_argument_value(internal_arg.value)
        else:
            for internal_arg in call_node.func.keywords:
                function_internal_params[internal_arg.arg] = self.get_argument_value(internal_arg.value)

        return function_internal_params

    def extract_function_external_params(self, call_node):
        external_arg = None
        if call_node.args != []:
            
            if isinstance(call_node.args[0], ast.Name):
                external_arg = call_node.args[0].id
            
            if isinstance(call_node.args[0], ast.List):
                external_arg = [self.get_argument_value(item) for item in call_node.args[0].elts]
                
            if isinstance(call_node.args[0], ast.Subscript):
                external_arg = call_node.args[0].value.id
        else:
            external_arg = 'None'

        return external_arg

    def get_code_ast_info(self, code):
        """
        Extract the key information of the code, including the function name Func_name, the called API name (func_name), the parameters (func_params), the external input value (external_arg), and the input variable name (variable_name)
        """
        global variable_name, func_name, func_params, external_arg, Func_name
        try:
            tree = ast.parse(code)
            code_ast_info = []
            for node in ast.walk(tree):
                
                if isinstance(node, ast.FunctionDef):
                    Func_name = node.name
                if isinstance(node, ast.Assign):
                    variable_name = self.extract_assign_variable_name(node)

                    if isinstance(node.value, ast.Call):

                        func_name = self.extract_func_name(node.value)
                        external_arg = self.extract_function_external_params(node.value)
                        if func_name:
                            func_params = self.extract_function_internal_params(node.value)
                            code_ast_info.append({'Function_name':Func_name,
                                                'Info':{"source_node": external_arg,
                                                "DL API": {func_name: func_params},
                                                "target_node": variable_name}})

                    if isinstance(node.value, ast.Name):
                        code_ast_info.append({'Function_name':Func_name,
                                            'Info':{"source_node": node.value.id,
                                            "DL API": {None: None},
                                            "target_node": variable_name}})

        except Exception as e:
            print('Error in parsing code: ', e)
            raise

        return code_ast_info

    def save_code_ast_info(self, code_ast_info, file_name, code_info_save_path):
        """
        Save the information parsed by ast for code as a json file
        """
        json_data = jsonpickle.encode(code_ast_info, indent=2)
        if not os.path.exists(code_info_save_path):
            os.makedirs(code_info_save_path)
        with open(code_info_save_path + '/' + file_name, 'w') as f: 
            f.write(json_data)