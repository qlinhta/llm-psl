from transformers import AutoModelForCausalLM


def get_other_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model
