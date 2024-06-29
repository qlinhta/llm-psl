import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, lora_alpha = 1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))
        self.scaling_factor = self.lora_alpha / self.rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling_factor


class LoRAEmbedding(nn.Module):
    def __init__(self, embedding, rank=4, lora_alpha = 1):
        super(LoRAEmbedding, self).__init__()
        self.embedding = embedding
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Parameter(torch.randn(embedding.embedding_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, embedding.embedding_dim))
        self.scaling_factor = self.lora_alpha / self.rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lora_embedding = embedded @ self.lora_A @ self.lora_B * self.scaling_factor
        return embedded + lora_embedding


def apply(model, rank=4):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_replace.append((name, module, LoRALayer(module.in_features, module.out_features, rank)))
        elif isinstance(module, nn.Embedding):
            modules_to_replace.append((name, module, LoRAEmbedding(module, rank)))

    for name, old_module, new_module in modules_to_replace:
        parent_module = model
        *path, last_name = name.split('.')
        for part in path:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, last_name, new_module)

    return model