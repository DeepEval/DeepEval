import os
import json
import importlib
import keras
import pandas as pd
from dynamic_checking_utils.get_task_name import get_task_name
from utils.df2excel import DataFrame2Excel
from utils.get_all_model_module_name import GetModelsFiles2Module
from utils.count_code_lines import count_code_lines


class Validation:
    """
    This class is used to verify the compilation of the DL model.
    """

    def __init__(self):
        pass

    def compile_code(self, model_lib):
        """
        Compile the code and return the verification information.
        """
        try:
            module = importlib.import_module(model_lib)
            model = module.dl_model()  # Note: dl_model()
            model.summary()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            compile_status = "compile success"
            compile_error = None
            error_type = None

        except Exception as error:
            compile_status = "compile failed"
            compile_error = error
            error_type = type(error).__name__  

        verification = {
            "model_lib": str(model_lib),
            "compile_status": str(compile_status),
            "error_type": str(error_type),
            "compile_error": str(compile_error),
        }
        return verification

    def batch_compile_models(self, models_path):
        """
        Batch compile the models and return the verification information.
        """
        model_lib_seqs = GetModelsFiles2Module(
            models_path, ".py", "DeepEval"
        ).get_processing_models_files()
        verifications = []

        for model_lib in model_lib_seqs:
            try:
                verification = self.compile_code(model_lib)
                verifications.append(verification)
                print(f"-----compile success: {model_lib} -----")

            except Exception as e:
                verifications.append({
                    "model_lib": str(model_lib),
                    "compile_status": "compile failed",
                    "compile_error": str(e),
                    })
                print(f"-----compile failure: {e} -----")

        return verifications

    def main(self, promptings, llms):
        
        global df_data, pd_rows
        for llm in llms:
            dfs_last_row, dfs_last_col = [], []
            for prompting in promptings:
                models_path = os.path.join(
                    common_root_path, f"response/DeepEval/{llm}/{prompting}"
                )
                verifications = self.batch_compile_models(models_path)

                save_json_path = os.path.join(
                    common_root_path,
                    f"evaluation/dynamic_checking/report/DeepEval/{llm}/{prompting}.json",
                )
                if not os.path.exists(os.path.dirname(save_json_path)):
                    os.makedirs(os.path.dirname(save_json_path))

                with open(save_json_path, "w") as f:  
                    json.dump(verifications, f, indent=2) 

                tasks_names = get_task_name(models_path)
                count_dict_list = []

                for task in tasks_names:  
                    valid_pyfiles = 0
                    for pyfile in os.listdir(os.path.join(models_path, task)):
                        if pyfile.endswith(".py"):
                            if count_code_lines(os.path.join(models_path, task, pyfile))!= 0:
                                valid_pyfiles += 1

                    count_dict = {
                        "Benchmark": task,
                        "success": 0,
                        "failed": 0,
                        "valid_pyfiles": 0,
                        "rate(%)": 0,
                    }

                    for item in verifications:
                        if task + "." in item["model_lib"]:
                            if item["compile_status"] == "compile success":
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
                Total["Benchmark"] = "Total"
                df_data = pd.concat([df_data, Total.to_frame().T], ignore_index=True)
                total_success = df_data.loc[df_data["Benchmark"] == "Total", "success"].values[0]
                total_rate = 100 * total_success / df_data.loc[df_data["Benchmark"] == "Total", "valid_pyfiles"]

                df_data.loc[df_data["Benchmark"] == "Total", "rate(%)"] = total_rate
                total_row = df_data.iloc[-1].copy() 
                total_row["Benchmark"] = str(prompting)
                dfs_last_row.append(total_row) 

                total_col = df_data.iloc[:, -1].copy()
                total_col.name = str(prompting)
                dfs_last_col.append(total_col)

            first_col = df_data.iloc[:, 0].copy()  
            dfs_last_col.insert(0, first_col) 

            # Save the summary of all promptings data as a sheet (summary)
            summary_df = pd.DataFrame(dfs_last_row)
            summary_df.reset_index(drop=True, inplace=True)

            save_result_excel_path = os.path.join(
                common_root_path,
                f"evaluation/dynamic_checking/report/DeepEval/{llm}/{llm}.xlsx",
            )
            DataFrame2Excel(summary_df, save_result_excel_path).df2excel(sheet_name="summary")



if __name__ == "__main__":
    common_root_path = "/your_local_path/DeepEval"

    promptings = ["zeroshot", "oneshot", "oneshotcot", "fewshot"]

    llms = ["gpt_4o",
            "gpt_4o_mini",
            "codegemma_7b_it", 
            "codellama_7b_instruct_hf",
            "deepseek_coder_v2_lite_instruct",
            "deepseek_v2_lite_chat",
            "gemma_2_9b_it",
            "meta_llama_3_1_8b_instruct"]

    compile_validation = Validation()
    compile_validation.main(promptings, llms,)
