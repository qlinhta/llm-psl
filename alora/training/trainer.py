import torch
import torch.optim as optim
from tqdm import tqdm
import logging


def train(model, dataloader, epochs, learning_rate, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = torch.nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=epoch_loss / (len(progress_bar)))

        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    return model
