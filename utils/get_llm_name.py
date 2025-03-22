import re


def get_llm_name(input):
    """
    Get the llm name according to the model_id, which is convenient for saving files
    """
    return re.sub(r"[^a-z0-9]+", "_", input.split("/")[-1].lower()).strip("_")

