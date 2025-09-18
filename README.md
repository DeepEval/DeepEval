# How Do Large Language Models Perform in Deep Learning Code Generation? An Empirical Study
This is the implementation repository of our research paper, "How Do Large Language Models Perform in Deep Learning Code Generation? An Empirical Study", currently under review at TOSEM.
<p align="left">
    <!-- <a href="https://arxiv.org/abs/2212.14834"><img src="https://img.shields.io/badge/arXiv-2212.14834-b31b1b.svg"> -->
    <a href="https://doi.org/10.5281/zenodo.17100624"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17100624.svg"> </a>
</p>

## Benchmark
DeepEval is constructed and saved the YAML format in ```./benchmark/DeepEval```. DeepEval consists of 100 DL programming tasks, each associated with a *Requirement* and a *Reference Code*. The *Requirement* provides a detailed task requirement. The *Reference Code* is a correct task implementation, serving as a reference for evaluating the generated code.

## Promptings:
We employ four prompting methods to LLMs on DeepEval, as detailed in ```./prompts/deepeval```.<br>
• Zero-shot prompting directly provides our task requirement to the LLMs without examples.<br>
• One-shot prompting includes an example with the form of <example task, example code> pair.<br>
• One-shot Chain-of-Thought prompting (oneshot-cot prompting) is a variant of one-shot prompting that generates a chain-of-thought
(CoT) for the example task. The prompting includes one example with the form of <example task, CoT, example code> triple.<br>
• Few-shot prompting includes multiple examples with the form of <example task, example code> pairs.
## Studied LLMs
| Type       | Model                                   | Size | Team     | Source | Access method   |
|------------|-----------------------------------------|------|----------|--------|-----------------|
| General LLM| GPT-4o                                  | -    | OpenAI   | Closed | Official API    |
|            | GPT-4o mini                             | -    | OpenAI   | Closed | Official API    |
|            | DeepSeek-V2-Lite-Chat                   | 16B  | DeepSeek | Open   | HuggingFace     |
|            | Gemma-2-9B-Instruct                     | 9B   | Google   | Open   | HuggingFace     |
|            | Llama-3.1-8B-Instruct                   | 8B   | Meta     | Open   | HuggingFace     |
| Code LLM   | DeepSeek-Coder-V2-Lite-Instruct         | 16B  | DeepSeek | Open   | HuggingFace     |
|            | Codellama-7B-Instruct                   | 7B   | Meta     | Open   | HuggingFace     |
|            | CodeGemma-7B-Instruct                   | 7B   | Google   | Open   | HuggingFace     |
## Results:
### RQ1: Benchmark Effectiveness: *How effective is DeepEval in benchmarking the ability of LLMs on DL code generation tasks?*
As shown in the following figure, for RQ1, we employ zero-shot prompting for code generation across HumanEval, ClassEval, MLEval and DeepEval, measuring performance with *Code Executability Rate*.

<img src="evaluation/dynamic_checking/presentation/RQ1.png" width="500"/> <br>
### RQ2: Code Syntax: *How do LLMs perform in generating DL code regarding static behavior?*
As shown in the following figure, for RQ2, we perform static checking using Pylint with *Score* of 1200 DL programs for each LLM.

<img src="evaluation/static_checking/presentation/RQ2.png" width="500"/><br>
### RQ3: Code Semantics: *How do LLMs perform in generating DL code regarding semantic accuracy?*
As shown in the following figure, for RQ3, we assess the semantic similarity of LLM-generated code to DeepEval’s reference code with *API Call Set Similarity* and *API Call Sequences Similarity*.

<img src="evaluation/semantic_checking/presentation/RQ3.png" width="600"/><br>
### RQ4: Code Executability: *How do LLMs perform in generating DL code regarding dynamic behavior?*
As shown in the following table, for RQ4, we evaluate the dynamic behavior of LLM-generated 1200 DL programs for each LLM on DeepEval with *Code Executability Rate*. 

| LLMs              | Zero-shot | One-shot | #Change  | Oneshot-cot | #Change  | &Change | Few-shot | #Change  |
|-------------------|-----------|----------|----------|-------------|----------|---------|----------|----------|
| CodeLlama-7B      | 15.15     | 22.15    | ↑46.19   | 23.67       | ↑56.22   | ↑6.86   | 21.02    | ↑38.73   |
| Llama-3.1-8B      | 21.33     | 32.33    | ↑51.57   | 32.00       | ↑50.02   | ↓1.02   | 34.00    | ↑59.40   |
| CodeGemma-7B      | 30.30     | 27.18    | ↓10.30   | 29.87       | ↓1.419   | ↑9.90   | 37.58    | ↑24.03   |
| Gemma-2-9B        | 21.00     | 30.67    | ↑46.05   | 29.67       | ↑41.24   | ↓3.26   | 33.00    | ↑57.14   |
| DeepSeekCoder-2-16B | 35.67    | 42.33    | ↑18.67   | 43.33       | ↑21.47   | ↑2.36   | 47.49    | ↑33.14   |
| DeepSeek-2-16B    | 20.00     | 21.33    | ↑6.650   | 21.00       | ↑5.000   | ↓1.55   | 23.33    | ↑16.65   |
| GPT-4o-mini       | 51.33     | 69.67    | ↑35.73   | 64.00       | ↑24.64   | ↑8.14   | 68.33    | ↑33.12   |
| GPT-4o            | 79.26*    | 83.67*   | ↑5.564   | 81.00*      | ↑2.612   | ↓2.80   | 79.00*   | ↓0.328   |
| **Average**       | **34.26** | **41.17**| **↑25.02** | **40.61**  | **↑24.98** | –     | **42.97*** | **↑32.73** |


<font size="4">For a comprehensive exploration of the findings, additional results and detailed analyses are presented in the subsequent sections of our paper.</font>

## Usage
Ensure you're using the right setup and following the proper directory structure to seamlessly evaluate deep learning code generation with our tool.
### 🛠️ Setup
#### 1. Repository Setup
Please clone the repository and install necessary dependencies:
```
git clone https://github.com/DeepEval/DeepEval.git
```
#### 2. Adding to Python's Module Path
To set up the project, you need to add the project directory to your `PYTHONPATH`. You can do this by running the following command:
```bash
export PYTHONPATH=$PYTHONPATH:/your_local_path/DeepEval
```
#### Code Generation
Please follow below steps to prepare environment and direct LLMs to generate code. <br>Note: to configure the benchmarks (DeepEval, HumanEval, and MLEval), their corresponding promptings, and the LLMs' id, please specify these details within the `run_hf.sh` and `run_api.sh`.
   | Step                    | Command                             |
   | ------------------------| -------------------------------------- |
   | 1. Create Environment   | `conda create -n DeepEval python=3.10` |
   | 2. Activate Environment | `conda activate DeepEval`            |
   | 3. Install Dependencies | `conda install environment_deepeval.yml`|
   | 4. Run HuggingFace Model| `bash run_hf.sh`                       |
   | 5. Run OpenAI Model     | `bash run_api.sh`                      |   
#### Code Evaluation
Please follow below steps to prepare environment and evaluate the generated code by LLMs.<br> 
Note 1: to ensure reproducibility and avoid library conflicts due to the diversity of generated code, we have prepared three distinct environments, each corresponding to one of the benchmarks: DeepEval, HumanEval, and MLEval.<br>
Note 2: To address false positives from Pylint, such as the message "Unable to import 'tensorflow.keras.layers'" (as reported in [issue1](https://github.com/microsoft/vscode-python/issues/10598) and [issue2](https://github.com/tensorflow/tensorflow/issues/26813)), we have developed a specialized plugin. Before running `python evaluation/static_checking/syntatic_checking_main.py`, install the plugin using the following command: `pip install evaluation/static_checking/plugin`.
   
- #### ​**DeepEval**
- Environment Configuration:
   ```bash
   conda create -n DeepEval python=3.10
   conda activate DeepEval
   conda install environment_deepeval.yml
- ​Command:
  ```python
  python evaluation/static_checking/syntatic_checking_main.py   # RQ2
  python evaluation/semantic_checking/semantic_checking_main.py   # RQ3
  python evaluation/dynamic_checking/dynamic_checking_deepeval_main.py   # RQ1 and RQ4
---

- #### ​**HumanEval**
- ​Environment Configuration:
  ```bash
   conda create -n HumanEval python=3.10
   conda activate HumanEval
   conda install environment_humaneval.yml
   ```
- ​Command:
   ```python
   python evaluation/dynamic_checking/dynamic_checking_humaneval_main.py   # RQ1
---

- #### ​**ClassEval**
- ​Environment Configuration:
  ```bash
   conda create -n ClassEval python=3.10
   conda activate ClassEval
   conda install environment_classeval.yml
   ```
- ​Command:
   ```python
   python evaluation/dynamic_checking/dynamic_checking_classeval_main.py   # RQ1
---

- #### ​**MLEval**
- ​Environment Configuration:
  ```bash
   conda create -n MLEval python=3.10
   conda activate MLEval
   conda install environment_mleval.yml
   ```
- ​Command:
  ```python
  python evaluation/dynamic_checking/dynamic_checking_mleval_main.py   # RQ1
  ```

<!-- | Benchmark                     | Environment Configuration                          |Command |
   | ------------------------| -------------------------------------- |----|
   | DeepEval |1.`conda create -n DeepEval python=3.10`<br>2.`conda activate DeepEval`<br>3.`conda install environment_deepeval.yml`|`python syntatic_checking_main.py` (RQ2);<br>`python semantic_checking_main.py` (RQ3);<br>`python dynamic_checking_deepeval_main.py` (RQ1 and RQ4)|
   | HumanEval | 1.`conda create -n HumanEval python=3.10`<br>2.`conda activate HumanEval`<br>3.`conda install environment_humaneval.yml`<br>|`python dynamic_checking_humaneval_main.py` (RQ1)|
   | MLEval  | 1.`conda create -n MLEval python=3.10`<br>2.`conda activate MLEval`<br>3.`conda install environment_mleval.yml`<br>|`dynamic_checking_mleval_main.py` (RQ1)| -->
## Implementation Details
As listed as follows, we adhere to the specified experimental settings throughout the process. Finally, 1200 DL programs are generated for each LLM under four different promptings on 100 DL code generation tasks in DeepEval.
| Parameter         | Configuration                                                     |
|------------------------|--------------------------------------------------------------------------|
| Sampling Method        | Nucleus Sampling                      |
| Temperature            | 0.8                                   |
| Top-p Value            | 0.95                                  |
| Hardware               | 2 × NVIDIA V100 GPUs                  |
| Repeat times           | 3                                     |

## Contributing
If you would like to contribute to DeepEval, please fork the repository and submit a pull request. We welcome all contributions!

## License
This repository is under [MIT](https://github.com/FudanSELab/ClassEval/blob/master/LICENSE) license. But the data is distributes through [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Cite
Please cite our tool if this work is helpful to you:

```bibtex
@misc{DeepEval,
  author       = {Xiangyue Ma and
                  Xiaoting Du and
                  Chenglong Li and
                  Jiangtao Meng and
                  Xiaoke Fang and
                  Wenjie Ding and
                  Zheng Zheng},
  title        = {Artifacts for "How Do Large Language Models Perform in Deep Learning Code Generation? An Empirical Study"},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17100625},
  url          = {https://doi.org/10.5281/zenodo.17100625}
}
```
<!-- ## Contact

For any questions or inquiries, please contact the project maintainer at [xiangy_ma@buaa.edu.cn)](xiangy_ma@buaa.edu.cn)). -->