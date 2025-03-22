import ast
import astor


class FunctionalProgrammingTransformer(ast.NodeTransformer):
    """
    This class is used to transform the code to functional programming style.
    """

    def __init__(self):
        self.function_names = set()

    def visit_Expr(self, node):

        if isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) \
                and isinstance(node.value.func.value, ast.Name) \
                and node.value.func.value.id == "model" \
                and node.value.func.attr == "compile":
            return node
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):

        self.function_names.add(node.name)
        self.generic_visit(node)

        return node

    def visit_Assign(self, node):

        self.in_assign = True
        result = self.generic_visit(node)
        self.in_assign = False

        return result

    def visit_Call(self, node):

        if hasattr(self, 'in_assign') and self.in_assign:
            if isinstance(node.func, ast.Name) and node.func.id in self.function_names:
                if node.args==[]:
                    new_call = ast.Call(func=ast.Call(func=node.func, args=[], keywords=[]),
                                        args=[node.keywords[0].value],
                                        keywords=[])
                    return new_call
                if node.keywords==[]:
                    new_call = ast.Call(func=ast.Call(func=node.func, args=[], keywords=[]),
                                        args=[node.args[0]],
                                        keywords=[])

                    return new_call
            return node
