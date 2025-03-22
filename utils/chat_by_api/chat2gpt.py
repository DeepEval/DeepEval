import openai


class Chat2GPTLLM:
    """
    Chat with a GPT model using OpenAI's API.
    Default API parameter configuration:
    1) max_tokens: If not specified by the user, it defaults to the maximum token count for the model
    2) temperature = 0.8
    3) top_p = 0.95
    """

    def __init__(self):
        pass

    def chat(self, prompting, model_id, temperature, top_p, max_tokens):

        api_key = "your-openai-api-key"
        openai.api_key = api_key

        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[{"role": "user", "content": prompting}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        completion = response.get("choices")[0]["message"]["content"]

        return completion

