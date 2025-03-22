PYTHON_FILE="./main.py"
python3 "$PYTHON_FILE" \
     --model_local_path "/your_local_path/huggingface/hub" \
     --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
     --benchmark "./benchmark/DeepEval/" \
     --prompting "DeepEval" \
     --repeats 3



