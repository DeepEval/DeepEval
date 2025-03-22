import os
import json
import pandas as pd
import subprocess
from tabulate import tabulate
from dynamic_checking_utils.get_task_name import get_task_name
from utils.df2excel import DataFrame2Excel
from utils.getallfiles import GetAllFiles
from utils.count_code_lines import count_code_lines


class Validation:
    """
    validate the code by compiling the code
    """
    def __init__(self):
        pass

    def compile_code(self, pyfile):
        
        result=subprocess.run(["python", pyfile], capture_output=True, text=True)

        if result.returncode==0:
            compile_status = "compile success"
            compile_error = None
        else:
            compile_status = "compile failed"
            compile_error = result.stderr or result.stdout

        verification = {"pyfile_path": str(pyfile),
                        "compile_status": str(compile_status),
                        "compile_error": str(compile_error)}
        
        return verification

    def batch_compile_codes(self, code_path):
        pyfile_paths = GetAllFiles(code_path, ".py").get_all_files_in_directory()
        verifications = []

        for pyfile_path in pyfile_paths:
            try:
                verification = self.compile_code(pyfile_path)
                verifications.append(verification)
                print(f'-----compile success:{verification} -----')

            except Exception as e:
                print(f'-----compile failed: {e} -----')

        return verifications

    def main(self, promptings, llms):
        global df_data, pd_rows
        for llm in llms:
            dfs_last_row ,dfs_last_col = [], []

            for prompting in promptings:
                
                code_path = os.path.join(common_root_path, f'response/HumanEval/{llm}/{prompting}')
                verifications = self.batch_compile_codes(code_path)

                save_json_path = os.path.join(common_root_path,f"evaluation/dynamic_checking/report/HumanEval/{llm}/{prompting}.json")

                if not os.path.exists(os.path.dirname(save_json_path)):
                    os.makedirs(os.path.dirname(save_json_path))
                with open(save_json_path, 'w') as f: 
                    json.dump(verifications, f, indent=2)

                tasks_names = get_task_name(code_path)
                count_dict_list = []

                for task_name in tasks_names: 
                    count_dict = {"Benchmark": task_name, "success": 0, "failed": 0, "valid_pyfiles": 0, "rate(%)": 0}

                    valid_pyfiles = 0
                    for pyfile in os.listdir(os.path.join(code_path, task_name)):
                        if pyfile.endswith('.py'):
                            if count_code_lines(os.path.join(code_path, task_name, pyfile)) != 0:
                                valid_pyfiles += 1

                    for item in verifications:

                        if task_name + '/' in item['pyfile_path']:  
                            if valid_pyfiles !=0:
                                if item['compile_status'] == 'compile success':
                                    count_dict["success"] += 1
                                else:
                                    count_dict["failed"] += 1

                    count_dict["valid_pyfiles"] = valid_pyfiles

                    if valid_pyfiles != 0:
                        count_dict["rate(%)"] = 100 * count_dict["success"] / valid_pyfiles
                    else:
                        count_dict["rate(%)"] = 0

                    count_dict_list.append(count_dict)

                df_data = pd.DataFrame(count_dict_list)
                Total = df_data.sum(axis=0)
                Total['Benchmark'] = 'Total'
                df_data = pd.concat([df_data, Total.to_frame().T], ignore_index=True)
                total_success = df_data.loc[df_data['Benchmark'] == 'Total', 'success'].values[0]

                total_rate = 100 * total_success / df_data.loc[df_data['Benchmark'] == 'Total', 'valid_pyfiles']

                df_data.loc[df_data['Benchmark'] == 'Total', 'rate(%)'] = total_rate 
                total_row = df_data.iloc[-1].copy()
                total_row['Benchmark'] = str(prompting)
                dfs_last_row.append(total_row)

                total_col = df_data.iloc[:, -1].copy()
                total_col.name = str(prompting)
                dfs_last_col.append(total_col)

            first_col = df_data.iloc[:, 0].copy()
            dfs_last_col.insert(0, first_col)

            summary_df = pd.DataFrame(dfs_last_row)
            summary_df.reset_index(drop=True, inplace=True)
            save_result_excel_path = os.path.join(common_root_path,
                            f"evaluation/dynamic_checking/report/HumanEval/{llm}/{llm}.xlsx")
            DataFrame2Excel(summary_df, save_result_excel_path).df2excel(sheet_name='summary')


if __name__ == '__main__':
    common_root_path = "/your_local_path/DeepEval"

    promptings = ['humaneval']

    llms = ["gpt_4o",
            "gpt_4o_mini", 
            "codegemma_7b_it", 
            "codellama_7b_instruct_hf",
            "deepseek_coder_v2_lite_instruct",
            "deepseek_v2_lite_chat",
            "gemma_2_9b_it",
            "meta_llama_3_1_8b_instruct"]

    compile_validation = Validation()
    compile_validation.main(promptings, llms)
