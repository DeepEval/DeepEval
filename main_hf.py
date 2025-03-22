import os
import random
import argparse
import torch
from utils.get_llm_name import get_llm_name
from utils.getallfiles import GetAllFiles
from utils.get_sample_taskid import get_sample_taskid
from utils.extract_humaneval_text import extract_text
from utils.read_json_benchmark import read_humaneval, read_mleval
from utils.read_yaml_benchmark import GetBenchmarkYamlfiles
from utils.chat_by_hf.chat2deepseek import Chat2DeepSeekLLM
from utils.chat_by_hf.chat2gemma import Chat2GemmaLLM
from utils.chat_by_hf.chat2llama import Chat2LlamaLLM
from prompts.humaneval import HumanEvalPromptDesigner
from prompts.mleval import MLEvalPromptDesigner
from prompts.deepeval.zeroshot import ZeroshotPromptDesigner
from prompts.deepeval.oneshot import OneshotPromptDesigner
from prompts.deepeval.oneshot_cot import OneshotCoTPromptDesigner
from prompts.deepeval.fewshot import FewshotPromptDesigner
from deepcode_generation import DLcodeGeneration


class MultiDLcodeGeneration:
    """
    For each benchmark task, design and use multiple prompting techniques to generate deep learning code multiple times, and save the code and chat logs in time and order.
    """

    def __init__(self, model_id, model_cache_path, data_dtype, temperature, top_p, max_new_tokens):
        """
        Paras:
        1.model_id: str, the LLM model id
        2.temperature: float, the temperature of the LLM
        3.top_p: float, the top_p of the LLM
        4.max_tokens: int, the max_tokens of the LLM
        """
        if "deepseek" in model_id:
            self.chat2llm = Chat2DeepSeekLLM
        elif "gemma" in model_id:
            self.chat2llm = Chat2GemmaLLM
        elif "llama" in model_id:
            self.chat2llm = Chat2LlamaLLM
        else:
            raise ValueError("The model_id is invalid, please check!")

        self.model, self.tokenizer = self.chat2llm().load_model(
            model_id, model_cache_path, data_dtype
        )
        self.llm_name = get_llm_name(model_id)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def generate_code(self, prompt_designer, task_requirement, task_id, count):
        """
        Paras:
        1.prompt_designer: class, the prompting method to be used
        2.task_requirement: the specific task requirements
        3.task_id: str, the task name
        4.count: int, mark the number of experiments
        """
        promting_name = prompt_designer.__name__.replace("PromptDesigner", "").lower()

        save_code_full_path = f"./response/{self.llm_name}/{promting_name}/{task_id}"
        save_log_full_path = f"./chat_logs/{self.llm_name}/{promting_name}/{task_id}"
        dlcode_generator = DLcodeGeneration(
            count_num=count,
            save_code_path=save_code_full_path,
            save_log_path=save_log_full_path,
        )

        prompt, response = dlcode_generator.prompt2llm(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_designer=prompt_designer,
            tasks_requirement=task_requirement,
            chat2llm=self.chat2llm(),
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        dlcode_generator.save(prompt, response)

    def multi_dlcode_generation(self, prompt_designers, benchmark_path, repeats_count):
        """
        Paras:
        1.prompt_designers: list, multiple prompting techniques
        2.benchmark_path: str, benchmark path
        3.repeats_count: int, repeat times
        """
        for prompt_designer in prompt_designers: 

            # for DeepEval
            if "DeepEval" in benchmark_path:
                get_all_files = GetAllFiles(directory=benchmark_path, file_type=".yaml")
                benchmark_files = get_all_files.get_all_files_in_directory()

                for yaml_file in benchmark_files: 
                    get_yaml = GetBenchmarkYamlfiles(yaml_file)
                    task_id = "/" + os.path.basename(yaml_file)[:-5] 
                    task= get_yaml.read_yaml_data()  
                    task_requirement = task['Requirement']
                    for count in range(1, repeats_count + 1):  
                        self.generate_code(prompt_designer, task_requirement, task_id, count)

            # for HumanEval
            if "HumanEval" in benchmark_path:

                get_all_files = GetAllFiles(directory=benchmark_path, file_type=".jsonl")
                benchmark_files = get_all_files.get_all_files_in_directory()

                tasks = read_humaneval(benchmark_files[0]) 
                for task in tasks: 
                    task_id = "/" + task["id"].replace("/", "_") 
                    task_requirement = extract_text(task["requirement"])
                    for count in range(1, repeats_count + 1):
                        self.generate_code(prompt_designer, task_requirement, task_id, count)

            # for MLEval
            if "MLEval" in benchmark_path:

                get_all_files = GetAllFiles(directory=benchmark_path, file_type=".json")
                benchmark_files = get_all_files.get_all_files_in_directory()
                tasks = read_mleval(benchmark_files[0])

                # Randomly select tasks
                # task_ids = [task["id"] for task in tasks]
                # selected_ids = random.sample(task_ids, 100)

                # The following two lines of code are used in subsequent experiments to maintain the consistency of sampling.
                selected_ids = get_sample_taskid("response/MLEval/gpt_4o")
                selected_tasks = [task for task in tasks if task["id"] in selected_ids]  

                for task in selected_tasks:
                    task_id = ("/" + "task_" + str(task["id"]))
                    task_requirement = task["requirement"]
                    for count in range(1, repeats_count + 1):
                        self.generate_code(prompt_designer, task_requirement, task_id, count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_local_path", type = str, help="The LLM local path")
    parser.add_argument("--model_id", type = str, help="The LLM id")
    parser.add_argument("--benchmark", type = str, help="The benchmark dataset path")
    parser.add_argument("--prompting", type = str, help="The prompting method")
    parser.add_argument("--repeats", type = int, help="The repeat times")

    args = parser.parse_args()

    multi_DLcode_Generator = MultiDLcodeGeneration(
        model_id=args.model_id,
        model_cache_path=args.model_local_path,
        data_dtype=torch.bfloat16,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=1500,
    ) 

    prompting_map = {
        "HumanEval": [HumanEvalPromptDesigner],
        "MLEval": [MLEvalPromptDesigner],
        "DeepEval": [ZeroshotPromptDesigner, OneshotPromptDesigner, OneshotCoTPromptDesigner,FewshotPromptDesigner]
    }
    args.prompting = prompting_map.get(args.prompting)

    multi_DLcode_Generator.multi_dlcode_generation(
        prompt_designers=args.prompting,
        benchmark_path=args.benchmark,
        repeats_count=args.repeats,
    )