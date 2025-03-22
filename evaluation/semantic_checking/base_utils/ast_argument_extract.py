import ast

def get_argument_value(arg):
    """
    Extract the specific parameter value according to different parameter data types, including strings, numbers, lists, tuples, dictionaries, attributes, function calls, etc.
    """
    if isinstance(arg, ast.NameConstant):
        return arg.value
    elif isinstance(arg, ast.Str):
        return arg.s
    elif isinstance(arg, ast.Num):
        return arg.n
    elif isinstance(arg, ast.List):
        return [get_argument_value(item) for item in arg.elts]
    elif isinstance(arg, ast.Tuple):
        return tuple([get_argument_value(item) for item in arg.elts])
    elif isinstance(arg, ast.Dict):
        return {get_argument_value(k): get_argument_value(v) for k, v in zip(arg.keys, arg.values)}
    elif isinstance(arg, ast.Attribute):
        return arg.attr
    elif isinstance(arg, ast.Call):
        return extract_function_call_arguments(arg)
    else:
        return None

def extract_function_call_arguments(node):
    arguments = []
    if isinstance(node, ast.Call):
        for keyword in node.keywords:
            arguments.append({keyword.arg: get_argument_value(keyword.value)})
    return arguments

function_calls = []
def extract_function_calls(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_name = node.func.id
        function_calls.append(node.func.id)


def visit_node(node):
    if isinstance(node, ast.FunctionDef):
        print(f"Function: {node.name}")
    elif isinstance(node, ast.Call):
        function_calls = extract_function_calls(node)
        arguments = extract_function_call_arguments(node)
        for arg in arguments:
            print(arg)

    for child_node in ast.iter_child_nodes(node):
        visit_node(child_node)
