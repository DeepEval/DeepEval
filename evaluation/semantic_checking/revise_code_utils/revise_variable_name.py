import ast
import astor

class ReviseNameTransformer(ast.NodeTransformer):
    """
    This class is used to revise the variable name.
    """
    def __init__(self):
        self.left_variable_counter = 1
        self.left_variable = set()
        self.variable_mapping = {}
        self.scope_stack = []

    def visit_FunctionDef(self, node):

        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Assign(self, node):
        if len(self.scope_stack) > 1:
            return self.generic_visit(node)

        if isinstance(node.value, ast.Call):
            for i, value_args in enumerate(node.value.args):
                if isinstance(value_args, ast.Name):
                    if value_args.id in self.variable_mapping:
                        node.value.args[i].id = self.variable_mapping[value_args.id]
                elif isinstance(value_args, ast.List):
                    for j, arg in enumerate(value_args.elts):
                        if isinstance(arg, ast.Name) and arg.id in self.variable_mapping:
                            value_args.elts[j].id = self.variable_mapping[arg.id]

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in self.left_variable:
                    new_id = target.id + '_' +str(self.left_variable_counter)
                    self.variable_mapping[target.id] = new_id
                    target.id = new_id
                    self.left_variable_counter += 1
                self.left_variable.add(target.id)

        return node

    def visit_Call(self, node):
        if len(self.scope_stack) > 1:
            return self.generic_visit(node)

        if isinstance(node.func, ast.Name):
            if node.func.id in self.variable_mapping:
                node.func.id = self.variable_mapping[node.func.id]
        return node
