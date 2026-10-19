"""Skip-gram architecture with Negative Sampling (SGNS) for Word2Vec."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip-gram with Negative Sampling (SGNS) model.

    Maintains two embedding matrices:
        - in_embeddings (W_in): Target / center word vectors.
        - out_embeddings (W_out): Context word vectors.

    Args:
        vocab_size: Total number of unique tokens in vocabulary (V).
        embed_dim: Dimensionality of word embeddings (D, typically 100-300).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # sparse=True: each step only touches the rows in the batch (use SparseAdam / SGD)
        # Target (center) embeddings: W_in
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # Context embeddings: W_out
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights according to Mikolov's paper.

        - W_in initialized uniformly in [-0.5 / embed_dim, 0.5 / embed_dim].
        - W_out initialized to zeros.
        """
        init_range = 0.5 / self.embed_dim
        nn.init.uniform_(self.in_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.out_embeddings.weight)

    def forward(
        self,
        center_words: torch.Tensor,
        context_words: torch.Tensor,
        negative_words: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Negative Sampling loss for a mini-batch.

        Loss formula:
            L = - log(sigma(u_o^T v_c)) - sum_{k=1}^K log(sigma(- u_{n_k}^T v_c))

        Args:
            center_words: LongTensor of shape (B,) containing center word IDs.
            context_words: LongTensor of shape (B,) containing true context word IDs.
            negative_words: LongTensor of shape (B, K) containing negative word IDs.

        Returns:
            torch.Tensor: Scalar loss averaged over the batch.
        """
        # 1. Lookup embeddings
        # v_c: (B, D)
        v_c = self.in_embeddings(center_words)

        # u_o: (B, D)
        u_o = self.out_embeddings(context_words)

        # u_neg: (B, K, D)
        u_neg = self.out_embeddings(negative_words)

        # 2. Positive score: dot product u_o^T v_c
        # pos_score shape: (B,)
        pos_score = torch.sum(v_c * u_o, dim=-1)
        # pos_loss = -log(sigma(pos_score))
        pos_loss = -F.logsigmoid(pos_score)

        # 3. Negative score: dot product u_neg^T v_c for each of the K samples
        # v_c.unsqueeze(1) has shape (B, 1, D)
        # Element-wise product with u_neg (B, K, D) summed over D gives (B, K)
        neg_score = torch.sum(v_c.unsqueeze(1) * u_neg, dim=-1)
        # neg_loss = -sum_{k} log(sigma(-neg_score))
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=-1)

        # 4. Total batch loss (mean across batch)
        total_loss = torch.mean(pos_loss + neg_loss)
        return total_loss

    def get_embeddings(self) -> torch.Tensor:
        """Return the center word embeddings W_in, shape (V, D)."""
        return self.in_embeddings.weight.detach()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print("--- 1. Kiểm tra khởi tạo mô hình SkipGramModel ---")
    vocab_size = 1000
    embed_dim = 100
    model = SkipGramModel(vocab_size=vocab_size, embed_dim=embed_dim)
    print(f"Khởi tạo model thành công:")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Embedding dimension: {embed_dim}")
    print(f"- Tổng số tham số: {sum(p.numel() for p in model.parameters()):,}")

    print("\n--- 2. Kiểm tra Forward pass và hàm Loss ---")
    batch_size = 4
    num_neg = 5

    dummy_center = torch.tensor([10, 25, 40, 80], dtype=torch.long)
    dummy_context = torch.tensor([12, 30, 45, 85], dtype=torch.long)
    dummy_neg = torch.randint(0, vocab_size, size=(batch_size, num_neg), dtype=torch.long)

    loss = model(dummy_center, dummy_context, dummy_neg)
    print(f"Batch loss tính được: {loss.item():.4f}")

    print("\n--- 3. Kiểm tra Backward pass (Gradient) ---")
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backward pass & Optimizer step thành công mượt mà!")

    print("\n🎉 Kiểm tra Bước 2.3 & 2.4 THÀNH CÔNG RỰC RỠ!")
