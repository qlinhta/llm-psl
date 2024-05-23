from transformers import LlamaForCausalLM


def get_llama_model(model_name):
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model
