from models.LLMs.gpt2 import get_gpt2_model
from models.LLMs.llama import get_llama_model
from models.LLMs.other import get_other_model


def load(model_name):
    if 'gpt2' in model_name:
        model = get_gpt2_model(model_name)
    elif 'llama' in model_name:
        model = get_llama_model(model_name)
    else:
        model = get_other_model(model_name)
    return model
