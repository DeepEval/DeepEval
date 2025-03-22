import ast
import astor
import traceback
from utils.read_python_code import ReadPythonCode
from evaluation.semantic_checking.extract_sub_main_code import ExtractMainSubCodes
from evaluation.semantic_checking.revise_code_utils.revise_functional_programming import FunctionalProgrammingTransformer
from evaluation.semantic_checking.revise_code_utils.revise_variable_name import ReviseNameTransformer
from evaluation.semantic_checking.revise_code_utils.revise_loop import FlattenLoopTransformer
from evaluation.semantic_checking.revise_code_utils.add_ending_node import AddOutputTensor, AddOutputLayer
from evaluation.semantic_checking.revise_code_utils.add_external_arg import AddExternalArg
from evaluation.semantic_checking.revise_code_utils.merge_revised_main_sub_codes import MergeRevisedMainSubCodes
from utils.save_code import save_code


class ReviseSave_Codes(ReadPythonCode):
    """
    1.Get python file, read code
    2.Preprocess the code, extract the main function code and subfunction code
    3.Use the method in FunctionCallTransformer to modify the code with AST:
    4.Merge the modified main function and subfunction code
    5.Save the modified code
    """

    def __init__(self, source_code_path, file_type):
        super().__init__(source_code_path, file_type)
        self.source_code_path = source_code_path

    def revise_main_code(self, code):
        """
        Modify the main_code code form through AST, so that when the custom function is called, it conforms to the "functional programming" paradigm
        """
        try:
            tree = ast.parse(code)
            transformer= FlattenLoopTransformer()
            modified_tree = transformer.visit(tree)
            modified_main_code = astor.to_source(modified_tree)

            new_tree = ast.parse(modified_main_code)
            transformer = FunctionalProgrammingTransformer()
            modified_tree = transformer.visit(new_tree)
            modified_main_code = astor.to_source(modified_tree)

            new_tree = ast.parse(modified_main_code)
            transformer = ReviseNameTransformer()
            modified_tree = transformer.visit(new_tree)
            modified_main_code = astor.to_source(modified_tree)

            new_tree = ast.parse(modified_main_code)
            transformer = AddOutputLayer()
            modified_tree = transformer.visit(new_tree)
            modified_main_code = astor.to_source(modified_tree)

        except Exception as e:
            print("error in revise_main_code:", e)
            raise

        return modified_main_code

    def revise_sub_code(self, code):
        """
        Modify the subfunction code, set an ending node to meet the requirements of data sequence extraction.
        """
        try:
            tree = ast.parse(code)
            transformer = ReviseNameTransformer()
            modified_tree = transformer.visit(tree)
            modified_sub_code = astor.to_source(modified_tree)

            new_tree = ast.parse(modified_sub_code)
            transformer = AddExternalArg()
            modified_tree = transformer.add_external_arg(new_tree)
            modified_sub_code = astor.to_source(modified_tree)

            new_tree = ast.parse(modified_sub_code)
            transformer = AddOutputTensor()
            modified_tree = transformer.add_output_tensor(new_tree)
            modified_sub_code = astor.to_source(modified_tree)

        except Exception as e:
            print("error in revise_sub_code:", e)
            raise

        return modified_sub_code

    def merge_main_sub_code(self, main_code, subfunction_codes):
        """
        Merge the modified main function and subfunction code
        """
        try:
            for subfunction_code in subfunction_codes:
                main_tree = ast.parse(main_code)
                subfunction_tree = ast.parse(subfunction_code)
                transformer = MergeRevisedMainSubCodes(subfunction_tree)
                main_tree = transformer.visit(main_tree)
                new_code = astor.to_source(main_tree)
                main_code = new_code
        except Exception as e:
            print("error in merge_main_sub_code:", e)
            raise

        return main_code

    def main(self, pyfile):

        code = self.read_source_code(pyfile)
        main_code, sub_codes = ExtractMainSubCodes().preprocess_sub_main_code(code)
        modified_main_code = self.revise_main_code(main_code)
        modified_sub_codes = []
        for sub_code in sub_codes:
            modified_sub_code = self.revise_sub_code(sub_code)
            modified_sub_codes.append(modified_sub_code)
        merged_main_sub_code = self.merge_main_sub_code(modified_main_code, modified_sub_codes)

        return modified_main_code, modified_sub_codes, merged_main_sub_code
