import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from mainab import CustomDataset, create_dataloader, clean_text


def generate_predictions_and_references(model_path, tokenizer_path, data_path, pred_file, ref_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    dataset = CustomDataset(data_path, is_train=False)
    dataloader = create_dataloader(dataset, tokenizer, batch_size=1)

    predictions, references = [], []
    progress_bar = tqdm(total=len(dataloader), desc="Generating predictions")
    for batch in dataloader:
        inputs, _, targets = batch
        with torch.no_grad():
            output = model.generate(inputs['input_ids'], max_new_tokens=60,
                                    attention_mask=inputs['attention_mask'],
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample=False,
                                    temperature=0.9,
                                    top_k=40)
        pred_texts = [tokenizer.decode(o, skip_special_tokens=True).split("Description:")[-1].strip() for o in output]
        pred_texts = [clean_text(text) for text in pred_texts]
        predictions.extend(pred_texts)
        references.extend(targets)
        progress_bar.update(1)
    progress_bar.close()

    with open(pred_file, 'w', encoding='utf-8') as pred_f, open(ref_file, 'w', encoding='utf-8') as ref_f:
        for pred, ref in zip(predictions, references):
            pred_f.write(pred + '\n')
            ref_f.write(ref + '\n')


generate_predictions_and_references('./saved_model', './saved_model', './data/test.txt', 'hypotheses.txt',
                                    'references.txt')
