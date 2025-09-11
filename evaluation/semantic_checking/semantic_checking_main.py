import os
import pandas as pd
from revise_extract_main import ReviseExtractCodeMain
from similarity import APISetSimilarity, APISequenceSetsSimilarity
from utils.getallfiles import GetAllFiles
from utils.count_code_lines import count_code_lines
from utils.df2excel import DataFrame2Excel


class FunctionalityChecking:
    """
    Check if two pieces of code have the same functionality
    """
    def __init__(self, save_flag, similarity_metric):
        self.save_flag = save_flag
        self.similarity_metric = similarity_metric
        self.ReviseExtractCodeMain = ReviseExtractCodeMain(source_code_path=None)
        self.APISetSimilarity = APISetSimilarity(metric=self.similarity_metric['api_set_similarity'])
        self.APISequenceSetsSimilarity = APISequenceSetsSimilarity(metric=self.similarity_metric['api_seq_similarity'])

    def main_single_task(self, ref_pyfile, gen_pyfiles):
        global api_sets_similarity_dict, api_seqs_sim_dict
        task_name = ref_pyfile.split('/')[-1].split('.')[0]
        ref_api_call_sets, ref_api_sequences, _ = self.ReviseExtractCodeMain.main_single_pyfile(ref_pyfile)

        api_set_sim_list, api_seqs_sim_list = [], []
        valid_pyfiles = 0

        for gen_pyfile in gen_pyfiles:
            print('-----------------------------------------------------------------------------')
            print("Processing gen file: ", gen_pyfile.replace(COMMENROOTPATH, '').lstrip('/').lstrip('\\'))
            try:
                code_lines = count_code_lines(gen_pyfile)
                if code_lines != 0: # if the code is not empty
                    valid_pyfiles += 1
                    gen_api_call_sets, gen_api_sequences, _ = self.ReviseExtractCodeMain.main_single_pyfile(gen_pyfile)

                    if gen_api_call_sets != []:
                        api_sets_similarity_dict = self.APISetSimilarity.compute_api_set_similarity(
                            reference=ref_api_call_sets,
                            candidate=gen_api_call_sets
                        )
                                                               
                        if any(value != 0 for value in api_sets_similarity_dict.values()): 
                            api_set_sim_list.append(api_sets_similarity_dict['similarity'])
                    else:
                        print("Failed in processing:", "API sets is []")  

                    if gen_api_sequences != []:
                        api_seqs_sim_dict = self.APISequenceSetsSimilarity.matrix_api_seq_similarity(reference=ref_api_sequences,   candidate=gen_api_sequences)
                        
                        if any(value != 0 for value in api_seqs_sim_dict.values()):
                            api_seqs_sim_list.append(api_seqs_sim_dict['similarity'])
                    else:
                        print("Failed in processing:", "API sequenses is []") 

                    if gen_api_call_sets != [] and gen_api_sequences != []:
                        print("Success in processing: ", gen_pyfile.replace(COMMENROOTPATH, '').lstrip('/').lstrip('\\'))

            except Exception as e:
                print('error:', e)
                print("Error in processing: ", gen_pyfile.replace(COMMENROOTPATH, '').lstrip('/').lstrip('\\'))


        def mean_of_similarity_list(data_list,valid_files):
            mean = sum(data_list) / valid_files if valid_files != 0 else 0
            return mean

        result_dict = {'benchmark_name': benchmark_name,
                        'result':
                        {
                            'api_sets_similarity': mean_of_similarity_list(api_set_sim_list,valid_pyfiles),
                            'valid data of api_sets_similarity': len(api_set_sim_list),
                            'api_seq_sets_similarity': mean_of_similarity_list(api_seqs_sim_list,valid_pyfiles),
                            'valid data of api_seq_sets_similarity': len(api_seqs_sim_list)
                        },
                        'valid pyfiles': valid_pyfiles,
                        }

        return result_dict

    def main_batch_benchmarks(self, ref_pyfiles_path, gen_pyfiles_path):

        get_all_files = GetAllFiles(ref_pyfiles_path, '.py')
        ref_pyfiles = get_all_files.get_all_files_in_directory()

        pd_cols = []
        average_row = []
        for ref_pyfile in ref_pyfiles:
            task = ref_pyfile.split('/')[-1].split('.')[0]
            for gen_last_path in os.listdir(gen_pyfiles_path): 
                if task == gen_last_path:
                    get_all_files = GetAllFiles(gen_pyfiles_path + '/' + gen_last_path, '.py')
                    gen_pyfiles = get_all_files.get_all_files_in_directory()

                    result_dict = self.main_single_task(ref_pyfile, gen_pyfiles)
                    col = {
                        'Benchmark': result_dict['benchmark_name'],
                        'API Set Similarity': round(100 * result_dict['result']['api_sets_similarity'], 2),
                        'Valid APISet Data': result_dict['result']['valid data of api_sets_similarity'],
                        'API Seqs Similarity': round(100 * result_dict['result']['api_seq_sets_similarity'], 2),
                        'Valid APISeqs Data': result_dict['result']['valid data of api_seq_sets_similarity'],
                        'Valid Pyfiles': result_dict['valid pyfiles'],
                    }
                    pd_cols.append(col)

        output_data = pd.DataFrame(pd_cols)
        for col in output_data.columns:
            if col not in ['Task']:
                average_row.append(output_data[col].mean())
            else:
                average_row.append('Average')
        output_data.loc[len(output_data)] = average_row

        return output_data

    def main(self, promptings, llms, benchmark):

        ref_root_path = os.path.join(COMMENROOTPATH, f"benchmark/{benchmark}/")

        for llm in llms:
            for prompting in promptings:
                gen_root_path = os.path.join(
                    COMMENROOTPATH, 
                    f"response/DeepEval/{llm}/{prompting}"
                )
                data = self.main_batch_benchmarks(
                    ref_root_path, 
                    gen_root_path
                )
                
                if save_flag['save_result']:
                    save_result_excel_path = os.path.join(
                        COMMENROOTPATH,
                        f"evaluation/semantic_checking/report/DeepEval/{llm}.xlsx"
                    )
                    df2excel_writer = DataFrame2Excel(data, save_result_excel_path)
                    df2excel_writer.df2excel(sheet_name=prompting)


if __name__ == '__main__':
    COMMENROOTPATH = "/your_local_path/DeepEval/" 
    save_flag = {"save_result":True}

    similarity_metric = {'api_set_similarity': 'set_based',
                        'api_seq_similarity': 'rouge_l'}

    promptings = ["zeroshot", "oneshot", "oneshotcot", "fewshot"]
    benchmark = "DeepEval"

    llms = ["gpt_4o",
            "gpt_4o_mini", 
            "codegemma_7b_it", 
            "codellama_7b_instruct_hf",
            "deepseek_coder_v2_lite_instruct",
            "deepseek_v2_lite_chat",
            "gemma_2_9b_it",
            "meta_llama_3_1_8b_instruct"]

    functionality_checking = FunctionalityChecking(save_flag=save_flag,similarity_metric=similarity_metric)
    functionality_checking.main(promptings, llms, benchmark)