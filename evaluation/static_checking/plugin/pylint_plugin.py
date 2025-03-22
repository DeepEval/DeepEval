from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class ImportRewriterChecker(BaseChecker):
    """
    Handle Pylint False Positive for 'tensorflow.keras' imports,
    Rewrite and delegate to Pylint to avoid False Positive like "No module named 'keras' in tensorflow"
    """
    __implements__ = IAstroidChecker

    name = "import-rewriter"
    msgs = {
        "C9001": (
            "False Positive for module of 'tensorflow' imports",
            "false-positive-tensorflow-import",
            "Pylint False Positive for 'tensorflow.keras' imports, automatically rewritten to 'keras'",
        ),
    }

    def visit_importfrom(self, node):
        """
        Handle Pylint False Positive for 'tensorflow.keras' imports";
        Example: message "No module named 'keras' in tensorflow".
        We rewrite and delegate to Pylint.
        """
        if node.modname.startswith("tensorflow.keras"):
            # False Positive for module of 'tensorflow' imports
            self.add_message("false-positive-tensorflow-import", node=node)
            # correct the import statement to 'keras'
            node.modname = node.modname.replace("tensorflow.keras", "keras")


def register(linter):
    linter.register_checker(ImportRewriterChecker(linter))