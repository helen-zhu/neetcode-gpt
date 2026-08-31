import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        # batch, seq_len, embedding_dim -> b, l, hidden
        q = self.query(embedded)
        k = self.key(embedded)
        v = self.value(embedded)

        # batch, seq_len (q), seq_len (k)
        q_k = q @ k.mT / torch.sqrt(torch.tensor(self.attention_dim))
        lower_triangular = torch.tril(torch.ones(q_k.shape[-1], q_k.shape[-1]))
        mask = (lower_triangular == 0)
        q_k = q_k.masked_fill(mask, float('-inf'))

        softmax_scores = nn.functional.softmax(q_k, dim=-1)
        final_scores = softmax_scores @ v
        return torch.round(final_scores, decimals=4)


