import ast
import astor


class MergeRevisedMainSubCodes(ast.NodeTransformer):
    def __init__(self, subfunction_tree):
        self.subfunction_tree = subfunction_tree

    def visit_FunctionDef(self, node):
        if node.name == self.subfunction_tree.body[0].name:
            return self.subfunction_tree
        return self.generic_visit(node)