import torch
import torch.nn as nn
from torch.nn import functional as F
from params import *
from dataset import train_data, val_data, encode, decode
from BigramLanguageModel import BigramLanguageModel
from TransformerLanguageModel import TransformerLanguageModel


torch.manual_seed(1337)

# example: in these 8 + 1 integers [18, 47, 56, 57, 58, 1, 15, 47, 58], given
# [18], want to predict 47; given [18, 47], want to predict 56; given [18, 47, 56],
# want to predict 57... there are 8 <context, target prediction character> pairs 
x = train_data[:BLOCK_SIZE]
y = train_data[1:BLOCK_SIZE + 1]
for t in range(BLOCK_SIZE):
    context = x[:t + 1]
    target = y[t]
    # print(f'when input is {context}, target is {target}')


def get_batch(split):
    '''
    Generate a small batch of data of inputs x and targets y;
    each block in the batch is independent of each other, i.e.,
    they don't interact
    '''
    data = train_data if split == 'train' else val_data
    # lower bound is 0 (inclusive), upper bound is len(data) - block_size (exclusive), size is (batch_size)
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) # a vector containing 4 random block indices
    # stack these 4 blocks vertically, as by default dim=0
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix]) # (batch_size, block_size)
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad() # telling pytorch that the execution of this code block doesn't require tracking gradients at all
def estimate_loss():
    out = {}
    model.eval() # set model to eval mode, which will disable dropout layers and change behavior of batchNorm layers
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() # extract the value of a singe-element tensor
        out[split] = losses.mean()
    model.train() # set back to train mode
    return out


xb, yb = get_batch('train')

# print out 1 block worth of data
for b in range(BATCH_SIZE): # batch dimension
    for t in range(BLOCK_SIZE): # time dimension
        context = xb[b, :t + 1]
        target = yb[b, t]
        # print(f'when input is {context.tolist()}, target is: {target}')
    

# model = BigramLanguageModel()
model = TransformerLanguageModel()
m = model.to(DEVICE)
logits, loss = model(xb, yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
for step in range(MAX_ITERS):

    # every once in a while evaluate the loss on train and val sets
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # clear the gradient stored in the model/optimizer from last step
    loss.backward() # gien this scalar loss, compute the gradient of this loss with respect to the model parameters through back propagation
    optimizer.step() # gradient descent to update the model parameters

    # print(loss.item())

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))