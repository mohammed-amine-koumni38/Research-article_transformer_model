# -*- coding: utf-8 -*-
"""
transformer_decoder-2.py
Transformer décodeur uniquement (modèle de langage causal) pour la génération de mots de passe.
Optimisé pour les jeux de données de 375k+ mots de passe.

Améliorations clés par rapport à v1 :
- Modèle plus grand : D_MODEL=512, 8 couches, FF=2048
- Échauffement par étape + planning de LR cosinus (critique pour la stabilité du transformer)
- Lissage des étiquettes (meilleure généralisation, réduit le surapprentissage)
- Génération parallèle vectorisée (50-100x plus rapide qu'une par une)
- Support AMP (précision mixte) pour un entraînement GPU plus rapide

Exemple d'utilisation :
python transformer_decoder-2.py --train train.txt --eval eval.txt --amp
"""

import argparse
import copy
import math
import random
import string
from collections import Counter
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm



#  HYPERPARAMÈTRES PAR DÉFAUT

BATCH_SIZE = 64
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 8
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
EPOCHS = 30
LR = 3e-4



#  ANALYSEUR D'ARGUMENTS

parser = argparse.ArgumentParser(description="Transformer décodeur uniquement pour la génération de mots de passe")

parser.add_argument("--train", type=str, default="train.txt", help="Chemin vers le fichier de mots de passe d'entraînement")
parser.add_argument("--eval", type=str, default="eval.txt", help="Chemin vers le fichier de mots de passe d'évaluation")

parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--patience", type=int, default=7, help="Patience pour l'arrêt anticipé")
parser.add_argument("--val_split", type=float, default=0.1, help="Ratio de séparation pour la validation depuis l'ensemble d'entraînement")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Étapes d'échauffement linéaire avant la décroissance cosinus")
parser.add_argument("--label_smoothing", type=float, default=0.1, help="Epsilon de lissage des étiquettes (0=désactivé)")
parser.add_argument("--amp", action="store_true", help="Utiliser la précision mixte automatique (CUDA uniquement)")

parser.add_argument("--d_model", type=int, default=D_MODEL)
parser.add_argument("--nhead", type=int, default=NHEAD)
parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
parser.add_argument("--dim_feedforward", type=int, default=DIM_FEEDFORWARD)
parser.add_argument("--dropout", type=float, default=DROPOUT)
parser.add_argument("--max_len", type=int, default=113, help="Longueur maximale de la séquence tokenisée incluant START/END")

parser.add_argument("--model_path", type=str, default="best_decoder2_model.pt")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--pad_loss_weight",
    type=float,
    default=-1.0,
    help="Si >=0, utilise la CE pondérée avec ce poids PAD ; si <0, ignore les cibles PAD via ignore_index",
)

args = parser.parse_args()



#  REPRODUCTIBILITÉ

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)



#  DISPOSITIF

def choose_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), "cuda-unavailable"

    arch_list = []
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            arch_list = []

    capability = torch.cuda.get_device_capability(0)
    capability_sm = f"sm_{capability[0]}{capability[1]}"

    if arch_list and capability_sm not in arch_list:
        gpu_name = torch.cuda.get_device_name(0)
        print(
            f"Warning: CUDA device {gpu_name} ({capability_sm}) is not supported by this torch build "
            f"({', '.join(arch_list)}). Falling back to CPU."
        )
        return torch.device("cpu"), "cuda-arch-mismatch"

    return torch.device("cuda"), None


device, device_note = choose_device()
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
elif device_note == "cuda-arch-mismatch":
    print("GPU: incompatible with current torch build; running on CPU.")

use_amp = args.amp and device.type == "cuda"
if args.amp and device.type != "cuda":
    print("Warning: --amp requires a compatible CUDA device, disabling.")


def _resolve_amp_backend():
    # Préférer torch.amp si disponible, sinon revenir aux APIs plus anciennes torch.cuda.amp.
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
        return (
            lambda: torch.amp.autocast("cuda"),
            lambda: torch.amp.GradScaler("cuda"),
            "torch.amp",
        )

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        return (
            lambda: torch.cuda.amp.autocast(),
            lambda: torch.cuda.amp.GradScaler(),
            "torch.cuda.amp",
        )

    return None, None, None


amp_autocast, amp_grad_scaler, amp_backend = _resolve_amp_backend()
if use_amp and amp_autocast is None:
    print("Warning: AMP backend not available in this torch build, disabling --amp.")
    use_amp = False



#  VOCABULAIRE (basé sur les données)

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"


def _scan_train_chars(filepath):
    chars = set()
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as _f:
            for _line in _f:
                chars.update(_line.strip())
    except OSError:
        pass
    return chars


special_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN]
_train_chars = _scan_train_chars(args.train)
vocab = special_tokens + sorted(_train_chars if _train_chars else string.printable)
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}
n_characters = len(vocab)

PAD_IDX = char_to_idx[PAD_TOKEN]
START_IDX = char_to_idx[START_TOKEN]
END_IDX = char_to_idx[END_TOKEN]

print(f"Vocabulary size: {n_characters} (built from training data — was 103 with string.printable)")



#  TRAITEMENT DES DONNÉES

def load_passwords(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def analyze_dataset(passwords, name):
    if not passwords:
        print(f"[{name}] Empty dataset")
        return

    lengths = [len(p) for p in passwords]
    has_upper = sum(1 for p in passwords if any(c.isupper() for c in p))
    has_digit = sum(1 for p in passwords if any(c.isdigit() for c in p))
    has_special = sum(1 for p in passwords if any(c in string.punctuation for c in p))

    print(f"\n[{name}] Dataset stats")
    print(f"  Size: {len(passwords):,}")
    print(
        f"  Lengths -> min: {min(lengths)}, max: {max(lengths)}, "
        f"median: {np.median(lengths):.1f}, mean: {np.mean(lengths):.2f}"
    )
    print(f"  With uppercase: {100.0 * has_upper / len(passwords):.2f}%")
    print(f"  With digits:    {100.0 * has_digit / len(passwords):.2f}%")
    print(f"  With special:   {100.0 * has_special / len(passwords):.2f}%")


def build_length_distribution(passwords):
    lengths = [len(p) for p in passwords if p]
    if not lengths:
        return np.array([8]), np.array([1.0])

    counts = Counter(lengths)
    values = np.array(sorted(counts.keys()), dtype=np.int32)
    probs = np.array([counts[v] for v in values], dtype=np.float64)
    probs /= probs.sum()
    return values, probs


def pad_password(pw, max_len):
    tokens = [START_TOKEN] + list(pw) + [END_TOKEN]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        if tokens[-1] != END_TOKEN:
            tokens[-1] = END_TOKEN
    tokens += [PAD_TOKEN] * (max_len - len(tokens))
    return tokens


def encode_password(tokens):
    return torch.tensor([char_to_idx.get(t, PAD_IDX) for t in tokens], dtype=torch.long)


def decode_password(indices):
    chars = [idx_to_char.get(i.item() if torch.is_tensor(i) else i, "") for i in indices]
    result = "".join(chars)
    result = result.replace(START_TOKEN, "").replace(END_TOKEN, "").replace(PAD_TOKEN, "")
    return result


def create_batch(passwords, max_len, device):
    padded = [pad_password(pw, max_len=max_len) for pw in passwords]
    encoded = [encode_password(pw) for pw in padded]
    return torch.stack(encoded).to(device)



#  MODÈLE (DÉCODEUR UNIQUEMENT)

class DecoderOnlyTransformer(nn.Module):
    """Transformer décodeur uniquement de style GPT utilisant TransformerEncoder + masque causal."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = self._build_positional_encoding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN : entraînement plus stable pour les modèles profonds
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)

    @staticmethod
    def _build_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    @staticmethod
    def generate_causal_mask(seq_len, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x):
        seq_len = x.size(1)

        emb = self.embedding(x)
        emb = emb + self.pos_encoding[:, :seq_len, :]
        emb = self.dropout(emb)

        causal_mask = self.generate_causal_mask(seq_len, x.device)
        padding_mask = x.eq(PAD_IDX)

        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.classifier(out)
        return logits


# ========================
#  PLANNING DU TAUX D'APPRENTISSAGE
# ========================
def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    """Échauffement linéaire pendant warmup_steps, puis décroissance cosinus jusqu'à min_lr_ratio * lr_de_base."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ========================
#  ENTRAÎNEMENT / ÉVALUATION
# ========================
def build_criterion(pad_loss_weight, label_smoothing=0.0):
    if pad_loss_weight < 0:
        return nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)

    class_weights = torch.ones(n_characters, device=device)
    class_weights[PAD_IDX] = pad_loss_weight
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def run_epoch(model, data, max_len, batch_size, criterion, optimizer=None, scheduler=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    num_batches = 0
    current_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0

    iterator = range(0, len(data), batch_size)
    if is_train:
        iterator = tqdm(iterator, desc="Train", leave=False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i in iterator:
            batch_passwords = data[i : i + batch_size]
            batch = create_batch(batch_passwords, max_len=max_len, device=device)
            inp = batch[:, :-1]
            target = batch[:, 1:]

            if is_train:
                optimizer.zero_grad()

            if scaler is not None:
                with amp_autocast():
                    logits = model(inp)
                    loss = criterion(logits.reshape(-1, n_characters), target.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inp)
                loss = criterion(logits.reshape(-1, n_characters), target.reshape(-1))
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            if is_train and scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl, current_lr


def train_model(
    model, train_data, val_data, max_len, epochs, batch_size, lr,
    criterion, model_path, patience, warmup_steps,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = amp_grad_scaler() if use_amp else None

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print("\n" + "=" * 70)
    print("DECODER-ONLY TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"Warmup steps: {warmup_steps} | Total steps: {total_steps:,}")
    amp_status = f"{use_amp} ({amp_backend})" if use_amp else "False"
    print(f"AMP: {amp_status} | Label smoothing: {args.label_smoothing}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)

        train_loss, train_ppl, last_lr = run_epoch(
            model, train_data, max_len=max_len, batch_size=batch_size,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        val_loss, val_ppl, _ = run_epoch(
            model, val_data, max_len=max_len, batch_size=batch_size,
            criterion=criterion, optimizer=None,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {last_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state_dict": best_state,
                    "vocab": vocab,
                    "config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                        "max_len": max_len,
                    },
                },
                model_path,
            )
            print(f"  -> Best model saved to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model





# ========================
#  PRINCIPAL
# ========================
def main():
    print("=" * 70)
    print("DECODER-ONLY TRANSFORMER PIPELINE")
    print("=" * 70)

    train_data = load_passwords(args.train)
    eval_data = load_passwords(args.eval)

    if not train_data:
        raise ValueError("Train dataset is empty.")
    if not eval_data:
        raise ValueError("Eval dataset is empty.")

    analyze_dataset(train_data, "TRAIN")
    analyze_dataset(eval_data, "EVAL")

    max_observed_len = max(chain((len(p) for p in train_data), (len(p) for p in eval_data))) + 2
    max_len = min(args.max_len, max_observed_len)
    print(f"\nUsing max_len={max_len} (observed+2={max_observed_len}, cap={args.max_len})")

    random.shuffle(train_data)
    split_idx = int((1.0 - args.val_split) * len(train_data))
    split_idx = min(max(split_idx, 1), len(train_data) - 1)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]

    print(f"Train split: {len(train_split):,} | Val split: {len(val_split):,}")

    model = DecoderOnlyTransformer(
        vocab_size=n_characters,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=max_len,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_criterion(args.pad_loss_weight, label_smoothing=args.label_smoothing)
    model = train_model(
        model,
        train_data=train_split,
        val_data=val_split,
        max_len=max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        criterion=criterion,
        model_path=args.model_path,
        patience=args.patience,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
