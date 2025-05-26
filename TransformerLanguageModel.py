import torch
import torch.nn as nn
from torch.nn import functional as F
from params import *

EMBEDDING_DIM = 32
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_DIM // NUM_HEADS


class Head(nn.Module):
    """
    One head of self-attention
    """
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, HEAD_SIZE, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, HEAD_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, HEAD_SIZE, bias=False)
        # to specify that this tensor is not trainable parameter, instead is a static mask 
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, HEAD_SIZE)
        # print(k)
        q = self.query(x) # (B, T, HEAD_SIZE)
        # Now compute attention scores, a.k.a. affinities
        # C**-0.5 is divding `wei` by the square root of HEAD_SIZE, which is a normalization technique
        # described in the original transformer paper
        # The reasoning is, say k and q are both standard Gaussian distributed (0 mean 1 variance), the variance
        # (k @ q.T).var() is roughly 16. After this normalization the variance is back at 1.
        # The reason why this is important is because softmax essentially coverages to one-hot
        # vector if the input has very positive or very negative values. Particularly at initialization,
        # since the model doesn't have knowledge about the English language yet, wei should be evenly
        # diffused so that softmax doesn't cause each node to only pay attention to 1 single other node,
        # so variance back at 1 is important.
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, HEAD_SIZE) @ (B, HEAD_SIZE, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), the weird [:T, :T] logic is probably just dereferencing the buffer?
        wei = F.softmax(wei, dim=-1)
        # print(wei)
        # perform the weighted aggregation of the values: each token becomes the average of all previous
        # tokens (including itself) in the same batch
        v = self.value(x) # (B, T, HEAD_SIZE)
        # print(f'wei.shape: {wei.shape}, v.shape: {v.shape}')
        # print(f'wei[0]: {wei[0]}, v[0]: {v[0]}')
        out = wei @ v # (B, T, T) @ (B, T, HEAD_SIZE) -> (B, T, HEAD_SIZE)
        # print(f'out.shape: {out.shape}')
        # print(f'out[0]: {out[0]}')

        return out
    

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(NUM_HEADS)]) # nn.ModuleList is neither inherently sequential or parallel, i.e., it all depends on how forward() is implemented
        self.proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # the heads are parallel, which is why Karpathy says multi-head is analogus to group convolutions, to focus on different aspects
        out = self.proj(out) # apply projection once, "project layer going back to the residual pathway?"
        return out
    

class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    The attention heads are for communication between tokens,
    the MLP here is for individual computation.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # this multiply-by-4 is from the original paper, as its inner ffwd layer dimension is 4 times the model dimension
            # essentially allows more computation at the side of the residual pathway
            nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            nn.ReLU(), # remember from 4787, this is element-wise
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM) # apply projection once, "project layer going back to the residual pathway?"
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()

    def forward(self, x):
        # skip connections (residual), for the purpose of gradient highway when the network gets deep
        # TODO: this should particularly help the model at the beginning, but need to double check why
        x = x + self.sa(x)
        x = x + self.ffwd(x) # skip connections
        return x


class TransformerLanguageModel(nn.Module):
    '''
    Concretely, BLOCK_SIZE determines the size of `wei`
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
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM) # num_embeddings (a.k.a. vocab size), embedding_dim (size of embedding vector)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)

        # option A (single attention head)
        self.sa_head = Head() # conceptually, the self-attention head does nothing but recomputing the embedding vector corresponding to each token

        # option B (multiple parallel attention heads, followed by linear computation)
        self.sa_heads = MultiHeadAttention() # i.e., 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward()

        # option C (multiple sequential blocks of mult-head attention, interchanged with linear computation)
        self.blocks = nn.Sequential(
            Block(),
            Block(),
            Block()
        )

        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE) # language model head, map embedding dimension back to vocab dimension for logits

    def forward(self, idx, targets=None):
        # print(f'[TransformerLanguageModel] idx.shape: {idx.shape}')
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, embedding_dim)
        # print(f'[TransformerLanguageModel] tok_emb.shape: {tok_emb.shape}')
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, embedding_dim)
        # print(f'[TransformerLanguageModel] pos_emb.shape: {pos_emb.shape}')
        x = tok_emb + pos_emb # pos_emb implicitly broadcasted to (B, T, embeeding_dim)
        # x = self.sa_heads(x) # apply one head of self-attention, (B, T, C)
        # x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
        '''
        Purely eval
        '''
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens; TODO: related to positional embedding, but why is this only needed here and not the bigram model??
            # Answer: for self.token_embedding_table(), the input can be any (B, ...), e.g., (B, T + 1) is also fine, as the output would just be
            # (B, T + 1, C). However since `pos_emb` is independent of `idx`, its shape is always (T, C); then `x = tok_emb + pos_emb` doesn't work
            idx_cond = idx[:, -BLOCK_SIZE:] # this line is totally addable to the bigram model, no behavior change at all.
            logits, loss = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=1) # softmax along each row
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), sampling
            idx = torch.cat((idx, idx_next), dim=1) # concat along each row, (B, T+1)
        return idx