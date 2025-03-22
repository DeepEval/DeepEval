import os
from utils.save_code import save_code
from evaluation.semantic_checking.revise_code_main import ReviseSave_Codes
from evaluation.semantic_checking.extract_sequences_utils.extract_code_info import Extract_Code_Info
from evaluation.semantic_checking.extract_sequences_utils.extract_sequences import ExtractSequences
from evaluation.semantic_checking.extract_sequences_utils.merge_flatten_api_sequences import MergeFlattenAPISequences


class ReviseExtractCodeMain:
    """
    1. Read the source code;
    2. Modify the source code through AST to meet certain forms (without modifying the function of the original code);
    3. Extract the key information code_info of the modified code and save it to a file;
    4. According to the code_info in step3, "flatten" the DNN model structure, extract the code API sequence, data sequence, <API sequence, data sequence>;
    5. Replace the custom function elements in the API sequence of the main function with the API sequence of the sub function, and generate all possible sequence combinations;
    6. Realize batch processing;
    """
    def __init__(self, source_code_path):
        self.source_code_path = source_code_path
        self.ReviseSave_Codes = ReviseSave_Codes(self.source_code_path, '.py')
        self.Extract_Code_Info = Extract_Code_Info()
        self.ExtractSequences = ExtractSequences()
        self.MergeFlattenAPISequences = MergeFlattenAPISequences(main_api_sequences_dict=None, sub_api_sequences_dict_lists=None)

    def main_single_pyfile(self, pyfile):
        modified_main_code, modified_sub_codes,merged_main_sub_code= self.ReviseSave_Codes.main(pyfile)
        modified_main_code_info = self.Extract_Code_Info.get_code_ast_info(modified_main_code)


        main_api_sequences = self.ExtractSequences.extract_api_sequence(modified_main_code_info,
                                                                        start_node=['None'],
                                                                        ending_node=['output', 'outputs','output_layer'])
        
        main_api_sequences= {'Function_name': modified_main_code_info[0]['Function_name'], 'API Sequences':main_api_sequences}

        sub_data_sequences_lists, sub_api_paras_sequences_lists, sub_api_sequences_lists = [], [], []

        for sub_code in modified_sub_codes:
            sub_code_info = self.Extract_Code_Info.get_code_ast_info(sub_code)
            sub_api_sequences = self.ExtractSequences.extract_api_sequence(sub_code_info,
                                                                        start_node=['input_tensor'],
                                                                        ending_node=['output_tensor'])
            
            sub_api_sequences= {'Function_name': sub_code_info[0]['Function_name'], 'API Sequences':sub_api_sequences}
            sub_api_sequences_lists.append(sub_api_sequences)

        merged_flatten_api_sequences = MergeFlattenAPISequences(main_api_sequences,sub_api_sequences_lists)
        api_sequences = merged_flatten_api_sequences.merge_api_sequences()
        api_sequences = merged_flatten_api_sequences.flatten_list(api_sequences)

        api_call_sets = self.ExtractSequences.extract_api_call_sets(code_info=modified_main_code_info)


        return api_call_sets, api_sequences, merged_main_sub_code

    def main_batch_pyfiles(self,save_flag, save_revised_code_path):

        pyfiles = ast_main.ReviseSave_Codes.get_pyfiles()
        for pyfile in pyfiles:
            api_call_sets, api_sequences, merged_main_sub_code= ast_main.main_single_pyfile(pyfile)

            if save_flag:
                save_code(merged_main_sub_code, 'revised_' + os.path.basename(pyfile), save_revised_code_path)
