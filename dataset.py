import torch
from params import VOCAB_SIZE

# corpus = 'shakespeare.txt'
corpus = 'sanguoyanyi.txt'

with open(corpus, 'r', encoding='utf-8') as f:
    text = f.read()

# print('length of dataset in characters: ', len(text))
chars = sorted(list(set(text)))
# print(''.join(chars))
# print(len(chars))
assert VOCAB_SIZE == len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars)} # encoder: take a string, output a list of integers
itos = { i:ch for i, ch in enumerate(chars)} # decoder: take a list of integers, output a string

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n] # list of integers
val_data = data[n:] # list of integers