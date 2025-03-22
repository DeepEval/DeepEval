PYTHON_FILE="./main_api.py"
python3 "$PYTHON_FILE" \
     --model_id "gpt-4o-mini" \
     --benchmark "./benchmark/DeepEval/" \
     --prompting "DeepEval" \
     --repeats 3