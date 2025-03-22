import ast

class AddExternalArg(ast.NodeTransformer):
    """
    Add external argument for function call
    """
    def __init__(self):
        pass

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'split':
            x_arg = None
            for keyword in node.keywords:
                if keyword.arg == 'x':
                    x_arg = keyword.value
                    break
            
            if x_arg is None:
                x_arg = node.args[0]
            
            new_call = ast.Call(
                func=node,
                args=[x_arg],
                keywords=[]
            )
            return new_call
        return node

    def add_external_arg(self, code):
        tree = ast.parse(code)
        transformer = AddExternalArg()
        modified_tree = transformer.visit(tree)

        return modified_tree

