import os
import pylint_plugin
from collections import OrderedDict
import pandas as pd
from pylint.lint import Run
from static_checking_utils.get_task_name import get_task_name
from static_checking_utils.report import Report
from utils.df2excel import DataFrame2Excel
from utils.getallfiles import GetAllFiles
from utils.count_code_lines import count_code_lines


class StaticAnalysis:
    """
    Use pylint to perform static analysis on the code generated by LLM (under certain rules), save it as a json file.
    And give a score, and finally generate a report.
    ref
    https://pylint.readthedocs.io/en/stable/tutorial.html
    """

    def __init__(self, checking_rule, save_flag):
        self.checking_rule = checking_rule
        self.save_flag = save_flag

    def define_rules(self):
        rules = {
            # The current checking unit is a subdirectory, and there is no need to check for duplicate code
            'default': ("--disable=duplicate-code,",), 
        }

        return rules

    def run(self, code_files, save_path):
        global module_dict_list
        rules = self.define_rules()
        msg_template = "{path}: [{line},{column}]: [{msg_id}]: {msg}" 
        pylint_args = [
            "--load-plugins","pylint_plugin"
            "--msg-template", msg_template,
            code_files, 
            *rules[self.checking_rule],
            "--reports=n"]

        if os.path.isdir(code_files):
            pylint_args.append("--recursive=y")
            all_files = GetAllFiles(
                directory=code_files,
                file_type='py'
            ).get_all_files_in_directory()

            module_dict_list = []
            for file in all_files:
                module_name = file.split('/')[-1].split('.')[0] 
                total_code_lines = count_code_lines(file)
                codeline_dict = {
                    "module_name": module_name, 
                    "total_code_lines": total_code_lines
                }
                module_dict_list.append(codeline_dict)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = save_path.split('/')[-2]

        if self.save_flag["save_json_flag"]:
            output_file_path_json = f'{save_path + filename}.json'
            pylint_args.append(f"--output-format=json:{output_file_path_json},colorized")

        # step1: run pylint
        print("-----Running pylint-----")
        Run(pylint_args, exit=False)
        print("-----Pylint finished, Begin Counting----")
        # Extract information from the saved json file and count the number of errors
        message_counts_socre = Report(save_path).report(module_dict_list)

        return message_counts_socre

    def main(self, promptings, llms):
        """
        main function
        """
        global df_data
        for llm in llms:
            dfs_last_row ,dfs_last_col = [], []
            for prompting in promptings:
                pd_cols = []
                experiment_batch_code_files = os.path.join(
                    COMMONROOTPATH,
                    f'response/DeepEval/{llm}/{prompting}'
                )
                task_names = get_task_name(path=experiment_batch_code_files)

                for task in task_names:
                    task_code_files = os.path.join(
                        COMMONROOTPATH,
                        f'response/DeepEval/{llm}/{prompting}/{task}'
                    )
                    save_path = os.path.join(
                        COMMONROOTPATH,
                        f'evaluation/static_checking/report/DeepEval/{llm}/{prompting}/{task}/'
                    )
                    count_result = self.run(
                        code_files=task_code_files, 
                        save_path=save_path
                    )

                    count_result = OrderedDict([("Benchmark", task)] + list(count_result.items()))
                    pd_cols.append(count_result)

                df_data = pd.DataFrame(pd_cols)
                Total = df_data.sum(axis=0)
                Total['Benchmark'] = 'Total'
                df_data = pd.concat([df_data, Total.to_frame().T], ignore_index=True)

                save_result_excel_path = os.path.join(
                    COMMONROOTPATH,
                    f"evaluation/static_checking/report/DeepEval/{llm}/{llm}.xlsx"
                )

                DataFrame2Excel(
                    df_data, 
                    save_result_excel_path
                ).df2excel(sheet_name=prompting)

                total_row = df_data.iloc[-1].copy() 
                total_row['Benchmark'] = str(prompting)
                dfs_last_row.append(total_row) 

                total_col = df_data.iloc[:, -1].copy()
                total_col.name = str(prompting)
                dfs_last_col.append(total_col)

            first_col = df_data.iloc[:, 0].copy()
            dfs_last_col.insert(0, first_col)

            summary = pd.DataFrame(dfs_last_col).T[:-1]
            save_result_excel_path = os.path.join(
                COMMONROOTPATH,
                f"evaluation/static_checking/report/DeepEval/{llm}/{llm}.xlsx"
            )

            DataFrame2Excel(summary, save_result_excel_path).df2excel(sheet_name='summary')
            print(f"Static analysis for {llm} is finished.")


if __name__ == '__main__':
    COMMONROOTPATH = "/your_local_path/DeepEval"

    checking_rule = 'default'
    save_flag = {"save_json_flag": True}
    
    promptings = ["zeroshot", "oneshot", "oneshotcot", "fewshot"]

    llms = ["gpt_4o",
            "gpt_4o_mini", 
            "codegemma_7b_it", 
            "codellama_7b_instruct_hf",
            "deepseek_coder_v2_lite_instruct",
            "deepseek_v2_lite_chat",
            "gemma_2_9b_it",
            "meta_llama_3_1_8b_instruct"]

    my_pylinter = StaticAnalysis(checking_rule=checking_rule, save_flag=save_flag)
    my_pylinter.main(promptings, llms)
