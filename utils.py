import torch


def stupid_self_attention():
    '''
    Simplest way to pay attention: each embedding vector becomes the average
    of all words in the prefix (including itself) in the block
    '''
    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b,:t + 1] # (t, C)
            xbow[b, t] = torch.mean(xprev, 0) # take mean over rows

def lower_triangular_self_attention():
    '''
    Implement self attention with lower triangular square matrix trick
    '''
    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)

    # Given a batch, the raw embedding is (T, C), and we want each row of this
    # to become the average of the rows above it. The tril matrix must be of shape
    # (T, T) because:
    #   1. Its second dim must be T for multiplication
    #   2. It must be square matrix to preserve the shape of (T, C)
    tril = torch.tril(torch.ones(T, T))
    wei = tril / torch.sum(a, 1, keepdim=True) # a now look like [[1, 0, 0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]]
    xbow2 = wei @ x # (T, T) @ (B, T, C) is implicitly broadcasted to (B, T, T) @ (B, T, C) => (B, T, C)
    # touch.allclose(xbow, xbow2)

def softmax_self_attention():
    '''
    Implement self attention with the trick but in a way that is aligned with training the transformer,
    i.e., which part is data dependent, which part is masked, and how the weights of each token within
    a batch is deduced through softmax
    '''
    B, T, C = 4, 8, 2 # batch, time, channels
    x = torch.randn(B, T, C)
    tril = torch.tril(torch.ones(T, T))
    # this is initialzing the affinity matrix, i.e., how much each token should be influenced
    # by other tokens in the same batch. This will be data dependent when we train the transformer
    wei = torch.zeros((T, T)) # wei is [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # this is saying that in a decoder transformer block, data from the past should not depend on data in the future
    wei = wei.masked_fill(tril == 0, float('-inf')) # wei is [[0, -inf, -inf], [0, 0, -inf], [0, 0, 0]]
    # this is using softmax to transform the affinity matrix into sum-to-1 weights for all tokens within each batch
    wei = F.softmax(wei, dim=-1) # softmax along each row: e^0 = 1, e^{-inf} = 0, which becomes [[1, 0, 0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]]
    
    xbow3 = wei @ x
    # touch.allclose(xbow, xbow3)