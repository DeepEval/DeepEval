from transformers import AutoTokenizer, AutoModelForCausalLM
hf_access_token = 'your_access_token_of_huggingface'

class Chat2LlamaLLM:
    """
    1.load_model: load model and tokenizer
    2.chat: interact with model, generate code
    """

    def __init__(self):
        pass

    def load_model(self, model_local_path, cache_dir, data_dtype):

        tokenizer = AutoTokenizer.from_pretrained(
            model_local_path, 
            token=hf_access_token,
            cache_dir=cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            device_map="auto",
            torch_dtype=data_dtype,
            token=hf_access_token,
            cache_dir=cache_dir,
        )

        return model, tokenizer

    def chat(self, model, tokenizer, prompting, temperature, top_p, max_new_tokens):

        chat = [{"role": "user", "content": prompting}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        generated_text = tokenizer.decode(outputs[0]) 
        response = generated_text[len(prompt):].strip() 


        return response

