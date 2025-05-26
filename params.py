import torch


# Hyperparameters
BATCH_SIZE = 32 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 8 # what is the maximum context length for predictions?
MAX_ITERS = 5000
# MAX_ITERS = 1
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
EVAL_ITERS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# VOCAB_SIZE = 65 # shakespeare
VOCAB_SIZE = 3974 # sanguoyanyi