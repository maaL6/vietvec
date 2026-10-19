"""Training loop for SkipGramModel: SparseAdam + linear learning-rate decay."""

import os
import time

import torch

from vietvec.dataset import SkipGramBatcher
from vietvec.model import SkipGramModel


def save_embeddings(model: SkipGramModel, words, path: str) -> None:
    """Save the center embeddings (W_in) with their words in a light .pt file."""
    torch.save({"words": list(words), "embeddings": model.get_embeddings().cpu()}, path)


def train(
    model: SkipGramModel,
    batcher: SkipGramBatcher,
    words,
    epochs: int = 5,
    lr: float = 2e-3,
    out_dir: str = "checkpoints",
    log_every: int = 2000,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    total_steps = epochs * batcher.estimate_batches()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(1e-4, 1 - step / total_steps))
    print(f"~{total_steps:,} steps total ({total_steps // epochs:,}/epoch)")

    step, running = 0, 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        for center, context, negatives in batcher:
            loss = model(center, context, negatives)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item() if step % 100 == 0 else 0.0  # .item() syncs the GPU; sample it
            step += 1
            if step % log_every == 0:
                print(
                    f"epoch {epoch} step {step:,} loss {running / (log_every / 100):.4f} "
                    f"lr {scheduler.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                    flush=True,
                )
                running = 0.0

        save_embeddings(model, words, os.path.join(out_dir, "vietvec.pt"))  # overwritten every epoch
        print(f"== epoch {epoch} done in {time.time() - t0:.0f}s", flush=True)
