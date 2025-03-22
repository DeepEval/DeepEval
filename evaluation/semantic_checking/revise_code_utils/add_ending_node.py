import ast
import astor


class AddOutputTensor():
    """
    Add a output_tensor variable in the end of the function
    """
    def __init__(self):
        pass

    def add_output_tensor(self, tree):
        global last_assignment_node
        last_variable_name = ""  
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):
                    last_assignment_node = node
                    last_variable_name = last_assignment_node.targets[0].id
                
        new_assignment_node = ast.Assign(
            targets=[ast.Name(id='output_tensor', ctx=ast.Store())],
            value=ast.Name(id=last_variable_name, ctx=ast.Load())
        )

        for idx, statement in enumerate(tree.body[0].body):
            if isinstance(statement, ast.Return):
                last_variable_name_value = astor.to_source(last_assignment_node.targets[0]).strip()
                
                if last_variable_name_value != 'output_tensor':
                    tree.body[0].body.insert(idx, new_assignment_node)
                break

        return tree

class AddOutputLayer(ast.NodeTransformer):
    """
    for the main function code, add an output_layer variable;
    Note that the flag is whether the value of the previous Assign node is the Model function, and whether there is an output_layer variable
    """
    def __init__(self):
        self.previous_assign = None
        self.output_layer_exists = False

    def visit_Assign(self, node):

        if isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'output_layer' or node.targets[0].id == 'output' or node.targets[0].id == 'outputs':
                self.output_layer_exists = True

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Model':
            if self.previous_assign and not self.output_layer_exists:
                previous_var = self.previous_assign.targets[0].id
                new_assign = ast.Assign(
                    targets=[ast.Name(id='output_layer', ctx=ast.Store())],
                    value=ast.Name(id=previous_var, ctx=ast.Load())
                    )
                return [new_assign, node]

        self.previous_assign = node

        return node

    def add_output_layer(self, code):
        tree = ast.parse(code)
        transformer = AddOutputLayer()
        modified_tree = transformer.visit(tree)

        return modified_tree
