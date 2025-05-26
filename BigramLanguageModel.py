import torch
import torch.nn as nn
from torch.nn import functional as F
from params import *


class BigramLanguageModel(nn.Module):
    '''
    Note that for bigram model, BLOCK_SIZE doesn't matter because the next token
    is only dependent on the previous token.
    '''
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # nn.Embedding() returns a function which takes in a tensor of indices of shape (batch_size, ...)
        # and outputs a tensor of shape (batch_size, ..., embedding_dim), where each index is replaced by
        # it's corresponding embedding vector
        # here, using vocab_size as embedding_dim makes sense because then the embedding vector is a
        # natural representation of probability distribution over vocab (ofc, after softmax); this is a
        # bigram language model because it directly maps the previous word to a distribution over the
        # next word, which doesn't involve any additional context.
        # With only this embedding layer as the model, the model is not even a linear transformation
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE) # num_embeddings (a.k.a. vocab size), embedding_dim (size of embedding vector)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C), logits means "raw values", a.k.a., pre-softmax

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # B = 4, T = 8, C = 65
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # input should be of shape (N = batch_size, num_of_classes), and the values should be raw logits
            # here we treat B * T = N
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=1) # softmax along each row
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), sampling
            idx = torch.cat((idx, idx_next), dim=1) # concat along each row, (B, T+1)
        return idx