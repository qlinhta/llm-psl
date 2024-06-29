import torch
import torch.optim as optim
from tqdm import tqdm
import logging


def train(model, dataloader, epochs, learning_rate, device, accumulation_steps=2):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)
            labels = torch.cat([labels[:, 1:], labels[:, :1]], dim=-1)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=epoch_loss / (i + 1))

        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    return model
