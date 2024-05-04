import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import os
import sys
import time
import math
import warnings

warnings.filterwarnings("ignore")


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGram, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.out_embed.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-0, 0)

    def forward(self, target, context, neg_context):
        emb_target = self.in_embed(target)
        emb_context = self.out_embed(context)
        emb_neg_context = self.out_embed(neg_context)

        score = torch.mul(emb_target, emb_context).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_context, emb_target.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))


class SkipGramTrainer:
    def __init__(self, model, data, batch_size, n_negs, lr, n_epochs, save_dir):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.n_negs = n_negs
        self.lr = lr
        self.n_epochs = n_epochs
        self.save_dir = save_dir
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        print("Training started")
        for epoch in range(self.n_epochs):
            total_loss = 0
            for i, (target, context, neg_context) in enumerate(self.data):
                self.optimizer.zero_grad()
                loss = self.model(target, context, neg_context)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if i % 1000 == 0:
                    print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, total_loss / (i + 1)))
            print("Epoch: {}, Total Loss: {}".format(epoch, total_loss))


class SkipGramDataLoader:
    def __init__(self, data, batch_size, n_negs):
        self.data = data
        self.batch_size = batch_size
        self.n_negs = n_negs
        self.vocab_size = len(self.data)
        self.id2word = {i: w for i, w in enumerate(self.data)}
        self.word2id = {w: i for i, w in enumerate(self.data)}
        self.word_freq = {w: 0 for w in self.data}
        self.get_word_freq()
        self.neg_sampling_dist = self.get_neg_sampling_dist()
        self.data = self.get_data()

    def get_word_freq(self):
        for w in self.data:
            self.word_freq[w] += 1

    def get_neg_sampling_dist(self):
        dist = np.array([self.word_freq[w] for w in self.data])
        dist = np.power(dist, 0.75)
        dist = dist / np.sum(dist)
        return dist

    def get_data(self):
        data = []
        for i in range(1, len(self.data) - 1):
            target = self.word2id[self.data[i]]
            context = [self.word2id[self.data[i - 1]], self.word2id[self.data[i + 1]]]
            neg_context = np.random.choice(self.vocab_size, self.n_negs, p=self.neg_sampling_dist)
            data.append((target, context, neg_context))
        return data

    def get_batch(self):
        random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i + self.batch_size]
            target = torch.tensor([b[0] for b in batch], dtype=torch.long)
            context = torch.tensor([b[1] for b in batch], dtype=torch.long)
            neg_context = torch.tensor([b[2] for b in batch], dtype=torch.long)
            yield target, context, neg_context


if __name__ == "__main__":
    set_random_seeds(42)

    data = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    BATCH_SIZE = 64
    N_NEGS = 5
    EMB_DIM = 100
    LR = 0.01
    N_EPOCHS = 10
    SAVE_DIR = "models"

    data_loader = SkipGramDataLoader(data, BATCH_SIZE, N_NEGS)
    model = SkipGram(data_loader.vocab_size, EMB_DIM)
    trainer = SkipGramTrainer(model, data_loader.get_batch(), BATCH_SIZE, N_NEGS, LR, N_EPOCHS, SAVE_DIR)
    trainer.train()
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "skipgram.pth"))