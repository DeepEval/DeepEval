import ast
import astor
from .revise_variable_name import ReviseNameTransformer
from .revise_functional_programming import FunctionalProgrammingTransformer


class FlattenLoopTransformer(ast.NodeTransformer):
    """
    This class is used to transform the code to flatten the loop.
    """
    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
            if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant):
                loop_count = node.iter.args[0].value
                loop_body = node.body
                new_body = loop_body * loop_count

                return new_body
        return node
