import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Chat2DeepSeekLLM:
    """
    1.load_model: load model and tokenizer
    2.chat: interact with model, generate code
    """
    def __init__(self):
        pass

    def load_model(self, model_local_path, cache_dir, data_type):

        tokenizer = AutoTokenizer.from_pretrained(
            model_local_path,
            cache_dir=cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            device_map="auto",
            torch_dtype=data_type,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        model.generation_config = GenerationConfig.from_pretrained(model_local_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return model, tokenizer

    def chat(self, model, tokenizer, prompting, temperature, top_p, max_new_tokens):
        
        chat = [{'role': 'user', 'content': prompting}]
        inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True, 
            return_tensors="pt").to("cuda")

        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        return response


