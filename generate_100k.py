# -*- coding: utf-8 -*-
"""
generate_100k.py
Générer 100 000 mots de passe depuis le point de contrôle du Transformer décodeur uniquement.

Utilisation :
  python generate_100k.py --model best_decoder2_model.pt
"""

import argparse
import math
import os
import random
import string
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


warnings.filterwarnings(
    "ignore",
    message=r"NVIDIA .* is not compatible with the current PyTorch installation.*",
    category=UserWarning,
)


parser = argparse.ArgumentParser(description="Générer 100 000 mots de passe depuis le modèle decoder2 (génération uniquement)")
parser.add_argument("--model", type=str, default="best_decoder2_model.pt")
parser.add_argument("--num", type=int, default=100000)
parser.add_argument("--output", type=str, default="100k_decoder2.txt")

parser.add_argument("--strategy", choices=["greedy", "nucleus", "beam"], default="nucleus")
parser.add_argument("--temperature", type=float, default=0.85)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--beam_width", type=int, default=5)
parser.add_argument("--max_gen_len", type=int, default=20)
parser.add_argument("--gen_batch_size", type=int, default=512)
parser.add_argument("--char_bias_strength", type=float, default=0.0)
parser.add_argument("--train", type=str, default="train.txt")

parser.add_argument("--match_eval_length", action="store_true", default=True)
parser.add_argument("--eval", type=str, default="eval.txt")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--flush_every", type=int, default=5000)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

args = parser.parse_args()


PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

all_characters = string.printable
special_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN]


def configure_vocab(vocab_override=None):
    global vocab, char_to_idx, idx_to_char, n_characters, PAD_IDX, START_IDX, END_IDX

    if vocab_override is None:
        selected_vocab = special_tokens + list(all_characters)
    else:
        selected_vocab = list(vocab_override)

    missing = [tok for tok in special_tokens if tok not in selected_vocab]
    if missing:
        raise ValueError(f"Checkpoint vocabulary is missing required special tokens: {missing}")

    vocab = selected_vocab
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_char = {i: c for i, c in enumerate(vocab)}

    n_characters = len(vocab)
    PAD_IDX = char_to_idx[PAD_TOKEN]
    START_IDX = char_to_idx[START_TOKEN]
    END_IDX = char_to_idx[END_TOKEN]


configure_vocab()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_password_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def build_length_distribution(passwords):
    lengths = [len(p) for p in passwords if p]
    if not lengths:
        return None, None
    counts = Counter(lengths)
    values = np.array(sorted(counts.keys()), dtype=np.int32)
    probs = np.array([counts[v] for v in values], dtype=np.float64)
    probs /= probs.sum()
    return values, probs


def build_char_freq_bias(passwords, device, strength=0.4):
    counter = Counter(c for pw in passwords for c in pw)
    total = max(1, sum(counter.values()))
    bias = torch.zeros(n_characters, device=device)
    for char, count in counter.items():
        if char in char_to_idx:
            bias[char_to_idx[char]] = strength * math.log(count / total + 1e-8)
    bias[PAD_IDX] = 0.0
    bias[START_IDX] = 0.0
    bias[END_IDX] = 0.0
    return bias


def decode_password(indices):
    chars = [idx_to_char.get(i, "") for i in indices]
    result = "".join(chars)
    return result.replace(START_TOKEN, "").replace(END_TOKEN, "").replace(PAD_TOKEN, "")


def choose_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("Warning: --device cuda requested, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
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
        return torch.device("cpu")
    return torch.device("cuda")


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout, norm_first=True):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = self._build_positional_encoding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

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
        emb = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        emb = self.dropout(emb)
        causal_mask = self.generate_causal_mask(seq_len, x.device)
        padding_mask = x.eq(PAD_IDX)
        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.classifier(out)


def nucleus_sample_batch(logits, temperature, top_p, char_bias=None):
    logits = logits.float()
    if char_bias is not None:
        logits = logits + char_bias.unsqueeze(0)
    logits = logits / max(temperature, 1e-4)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits[remove] = float("-inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1).squeeze(-1)
    return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def generate_parallel_batch(model, device, num_passwords, max_length, strategy, temperature, top_p, gen_batch_size=512, char_bias=None):
    model.eval()
    all_passwords = []
    with torch.no_grad():
        for start in range(0, num_passwords, gen_batch_size):
            n = min(gen_batch_size, num_passwords - start)
            seqs = torch.full((n, 1), START_IDX, dtype=torch.long, device=device)
            finished = torch.zeros(n, dtype=torch.bool, device=device)
            for step in range(max_length):
                if finished.all():
                    break
                logits = model(seqs)[:, -1, :].float()
                logits[:, PAD_IDX] = float("-inf")
                if step == 0:
                    logits[:, END_IDX] = float("-inf")
                if strategy == "greedy":
                    biased = logits + char_bias.unsqueeze(0) if char_bias is not None else logits
                    next_tokens = torch.argmax(biased / max(temperature, 1e-4), dim=-1)
                else:
                    next_tokens = nucleus_sample_batch(logits, temperature, top_p, char_bias=char_bias)
                next_tokens = next_tokens.masked_fill(finished, PAD_IDX)
                finished = finished | (next_tokens == END_IDX)
                seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)
            for i in range(n):
                tokens = seqs[i, 1:].tolist()
                chars = []
                for t in tokens:
                    if t in (END_IDX, PAD_IDX):
                        break
                    chars.append(idx_to_char.get(t, ""))
                all_passwords.append("".join(chars))
    return all_passwords


def generate_password_beam(model, device, max_steps, beam_width, temperature):
    beams = [([START_IDX], 0.0, False)]
    with torch.no_grad():
        for _ in range(max_steps):
            candidates = []
            all_finished = True
            for tokens, score, finished in beams:
                if finished:
                    candidates.append((tokens, score, True))
                    continue
                all_finished = False
                inp = torch.tensor([tokens], device=device)
                logits = model(inp)[0, -1, :].clone()
                logits[PAD_IDX] = float("-inf")
                logits = logits / max(temperature, 1e-4)
                log_probs = torch.log_softmax(logits, dim=-1)
                top_k = min(beam_width, log_probs.size(0))
                top_vals, top_idx = torch.topk(log_probs, k=top_k)
                for logp, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    candidates.append((tokens + [idx], score + float(logp), idx == END_IDX))
            if all_finished:
                break
            candidates.sort(key=lambda x: x[1] / max(1.0, (len(x[0]) - 1) ** 0.7), reverse=True)
            beams = candidates[:beam_width]
    best = max(beams, key=lambda x: x[1] / max(1.0, (len(x[0]) - 1) ** 0.7))
    return decode_password(best[0][1:])


def main():
    set_seed(args.seed)

    device = choose_device(args.device)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu")

    checkpoint_vocab = checkpoint.get("vocab") if isinstance(checkpoint, dict) else None
    configure_vocab(checkpoint_vocab)
    if checkpoint_vocab is not None:
        print(f"Using checkpoint vocabulary ({n_characters} tokens).")
    else:
        print(f"Warning: checkpoint has no saved vocabulary, using fallback printable vocabulary ({n_characters} tokens).")

    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    d_model = cfg.get("d_model", 256)
    nhead = cfg.get("nhead", 8)
    num_layers = cfg.get("num_layers", 6)
    dim_feedforward = cfg.get("dim_feedforward", 1024)
    dropout = cfg.get("dropout", 0.1)
    max_len = cfg.get("max_len", 113)
    norm_first = cfg.get("norm_first", True)

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = DecoderOnlyTransformer(
        vocab_size=n_characters, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        max_len=max_len, dropout=dropout, norm_first=norm_first,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(
        f"Model loaded. Config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
        f"ff={dim_feedforward}, max_len={max_len}, norm_first={norm_first}, vocab_size={n_characters}"
    )

    length_values, length_probs = None, None
    if args.match_eval_length and os.path.exists(args.eval):
        eval_data = read_password_lines(args.eval)
        length_values, length_probs = build_length_distribution(eval_data)
        print("Using eval length distribution." if length_values is not None else "Eval file empty, using fixed max_gen_len.")

    char_bias = None
    if args.char_bias_strength > 0 and os.path.exists(args.train):
        train_data = read_password_lines(args.train)
        char_bias = build_char_freq_bias(train_data, device, strength=args.char_bias_strength)
        print(f"Character frequency bias enabled (strength={args.char_bias_strength})")

    start_count = 0
    write_mode = "w"
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8", errors="ignore") as f:
            start_count = sum(1 for line in f if line.strip())
        if start_count >= args.num:
            print(f"Output already has {start_count:,} lines. Nothing to do.")
            return
        write_mode = "a"
        print(f"Resuming from {start_count:,}/{args.num:,}")

    remaining = args.num - start_count
    print(f"Generating {remaining:,} passwords...")
    print(f"Strategy: {args.strategy} | Temperature: {args.temperature} | top_p: {args.top_p} | batch_size: {args.gen_batch_size}")

    if args.strategy == "beam":
        generated_count = start_count
        with open(args.output, write_mode, encoding="utf-8") as f:
            with tqdm(total=args.num, initial=start_count, desc="Generating (beam)") as pbar:
                for _ in range(remaining):
                    target_length = int(np.random.choice(length_values, p=length_probs)) if length_values is not None else args.max_gen_len
                    pw = generate_password_beam(model, device, max_steps=min(target_length, max_len - 1), beam_width=args.beam_width, temperature=args.temperature)
                    f.write(pw + "\n")
                    generated_count += 1
                    if generated_count % args.flush_every == 0:
                        f.flush()
                    pbar.update(1)
    else:
        if length_values is not None:
            target_lengths = np.random.choice(length_values, size=remaining, p=length_probs).tolist()
            buckets = defaultdict(list)
            for idx, tl in enumerate(target_lengths):
                buckets[int(tl)].append(idx)
            all_passwords = [""] * remaining
            for tl, indices in tqdm(buckets.items(), desc="Generating by length"):
                batch_pws = generate_parallel_batch(
                    model, device, len(indices), max_length=tl,
                    strategy=args.strategy, temperature=args.temperature, top_p=args.top_p,
                    gen_batch_size=args.gen_batch_size, char_bias=char_bias,
                )
                for idx, pw in zip(indices, batch_pws):
                    all_passwords[idx] = pw
        else:
            all_passwords = generate_parallel_batch(
                model, device, remaining, max_length=args.max_gen_len,
                strategy=args.strategy, temperature=args.temperature, top_p=args.top_p,
                gen_batch_size=args.gen_batch_size, char_bias=char_bias,
            )

        with open(args.output, write_mode, encoding="utf-8") as f:
            for pw in all_passwords:
                f.write(pw + "\n")

    print(f"Saved {args.num:,} passwords to {args.output}")


if __name__ == "__main__":
    main()
