import os
from pathlib import Path
from typing import Dict, Iterable
from datetime import datetime

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.midi_dataset import MidiSequenceDataset
from src.data.midi_tokenizer import VOCAB_SIZE
from src.models.transformer_gpt import CausalTransformerLM
from src.utils.positional_encoding import SinusoidalPositionalEncoding
from src.utils.seed import set_seed


# Logging and small utilities
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Config helpers
def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def to_int(x, default=None):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def to_float_tuple(xs: Iterable, default=None):
    try:
        return tuple(float(v) for v in xs)
    except Exception:
        return default


# Training helpers
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: nn.Module, path: str) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def lr_warmup_cosine(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1 + torch.cos(torch.tensor(progress * 3.1415926))).item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    step: int,
    max_steps: int,
    warmup: int,
    base_lr: float,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    log(f"start train pass | batches={len(loader)} | start_step={step} | max_steps={max_steps}")

    for i, (x, y) in enumerate(loader):
        if step >= max_steps:
            break

        x = x.to(device)
        y = y.to(device)

        T = x.size(1)
        mask = model.causal_mask(T, device)
        logits = model(x, tgt_mask=mask)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        cur_lr = lr_warmup_cosine(step, warmup, max_steps, base_lr)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        optim.step()

        total_loss += loss.item()

        if i % 50 == 0:
            tqdm.write(
                f"[train] step={step} batch={i}/{len(loader)} loss={loss.item():.4f} lr={cur_lr:.6f}"
            )

        step += 1

    avg = total_loss / max(1, len(loader))
    log(f"end train pass   | avg_loss={avg:.4f} | end_step={step}")
    return avg, step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0

    log(f"start eval pass  | batches={len(loader)}")
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        T = x.size(1)
        mask = model.causal_mask(T, device)
        logits = model(x, tgt_mask=mask)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
        count += 1
        if i % 50 == 0:
            tqdm.write(f"[eval ] batch={i}/{len(loader)} loss={loss.item():.4f}")
    avg = total / max(1, count)
    log(f"end eval pass    | avg_loss={avg:.4f}")
    return avg


# Main
def main():
    cfg = load_config("config/config.yaml")
    set_seed(to_int(cfg.get("seed", 42), 42))

    device = get_device()
    log("Loaded config")
    log(f"Device: {device}")

    # Coerce common params
    seq_len = to_int(cfg.get("seq_len", 512), 512)
    min_tokens = to_int(cfg.get("min_tokens", 256), 256)
    batch_size = to_int(cfg.get("batch_size", 16), 16)
    num_workers = to_int(cfg.get("num_workers", 2), 2)

    d_model = to_int(cfg.get("d_model", 512), 512)
    nhead = to_int(cfg.get("nhead", 8), 8)
    num_layers = to_int(cfg.get("num_layers", 6), 6)
    dim_ff = to_int(cfg.get("dim_feedforward", 2048), 2048)
    dropout = to_float(cfg.get("dropout", 0.1), 0.1)

    lr = to_float(cfg.get("lr", 3e-4), 3e-4)
    betas = to_float_tuple(cfg.get("betas", [0.9, 0.95]), (0.9, 0.95))
    weight_decay = to_float(cfg.get("weight_decay", 0.01), 0.01)
    max_steps = to_int(cfg.get("max_steps", 2000), 2000)
    warmup_steps = to_int(cfg.get("warmup_steps", 100), 100)
    grad_clip = to_float(cfg.get("grad_clip", 1.0), 1.0)
    val_interval = to_int(cfg.get("val_interval", 200), 200)
    vocab_size = to_int(cfg.get("vocab_size", VOCAB_SIZE), VOCAB_SIZE)

    # Dataset
    log(f"Preparing dataset from {cfg['midi_dir']} (seq_len={seq_len}, min_tokens={min_tokens})")
    ds = MidiSequenceDataset(cfg["midi_dir"], seq_len=seq_len, min_tokens=min_tokens)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    ds_tr, ds_va = random_split(ds, [n_train, n_val])
    log(f"Dataset total={len(ds)} | train={len(ds_tr)} val={len(ds_va)}")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log(f"Dataloaders ready | batch_size={batch_size} workers={num_workers}")

    # Model
    model = CausalTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
    ).to(device)
    model.pos = SinusoidalPositionalEncoding(d_model).to(device)

    n_params = count_params(model)
    log(f"Model built      | params={n_params:,} d_model={d_model} layers={num_layers} nhead={nhead}")

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    log(f"Optimizer ready  | AdamW(lr={lr}, betas={betas}, weight_decay={weight_decay})")
    log(f"Schedule         | warmup_steps={warmup_steps} max_steps={max_steps} grad_clip={grad_clip}")

    # Training loop
    best_val = float("inf")
    step = 0
    pbar = tqdm(total=max_steps, desc="training", dynamic_ncols=True)

    log("Begin training loop")
    while step < max_steps:
        tr_loss, step = train_one_epoch(
            model,
            dl_tr,
            optim,
            device,
            grad_clip=grad_clip,
            step=step,
            max_steps=max_steps,
            warmup=warmup_steps,
            base_lr=lr,
        )
        # Advance pbar approximately one epoch's worth, but do not exceed total
        pbar.update(min(max_steps - pbar.n, len(dl_tr)))

        if step % val_interval == 0:
            va_loss = evaluate(model, dl_va, device)
            log(f"Validation       | step={step} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
            if va_loss < best_val:
                best_val = va_loss
                best_path = cfg.get("best_ckpt", "artifacts/models/best.pt")
                Path(best_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)
                log(f"Saved best model | path={best_path} val_loss={va_loss:.4f}")

    pbar.close()
    log("Training finished")


if __name__ == "__main__":
    main()
