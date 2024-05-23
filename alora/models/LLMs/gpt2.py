from transformers import GPT2LMHeadModel


def get_gpt2_model(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model
