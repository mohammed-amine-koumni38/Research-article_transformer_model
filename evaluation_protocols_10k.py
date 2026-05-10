# -*- coding: utf-8 -*-
"""
Protocoles d'évaluation utilisés par les scripts de génération.
Séparé du code du modèle pour garder les fichiers d'entraînement/génération plus propres.

Exemples d'utilisation :
    python evaluation_protocols_10k.py --generated 10k_decoder2.txt --name Decoder2-10k
    python evaluation_protocols_10k.py --generated 10k_gru.txt --name GRU --protocol both
"""

import argparse
from collections import Counter
from pathlib import Path
import numpy as np


def normalize_for_coverage(pw):
    # Traiter uniquement un backslash final comme optionnel pour la correspondance de couverture.
    return pw[:-1] if pw.endswith("\\") else pw


def protocol_1_coverage_analysis(generated_passwords, test_data, name="Generated"):
    print("\n" + "=" * 70)
    print(f"PROTOCOLE 1: COVERAGE ANALYSIS - {name}")
    print("=" * 70)

    generated_set = set(generated_passwords)
    test_set = set(test_data)
    generated_set_norm = {normalize_for_coverage(pw) for pw in generated_passwords}
    test_set_norm = {normalize_for_coverage(pw) for pw in test_data}
    num_generated = len(generated_passwords)
    num_unique_generated = len(generated_set)
    num_test = len(test_data)

    hits_exact = generated_set.intersection(test_set)
    hit_rate_exact = len(hits_exact) / max(1, num_test) * 100
    hits_norm_count = len(generated_set_norm.intersection(test_set_norm))
    hit_rate_norm = hits_norm_count / max(1, num_test) * 100

    print(f"Generated: {num_generated:,} | Unique: {num_unique_generated:,} | Test: {num_test:,}")
    print(f"Unique ratio: {num_unique_generated / max(1, num_generated) * 100:.2f}%")
    print(f"Passwords found (exact): {len(hits_exact)}/{num_test} | Hit rate: {hit_rate_exact:.2f}%")
    print(f"Passwords found (backslash-aware): {hits_norm_count}/{num_test} | Hit rate: {hit_rate_norm:.2f}%")

    generated_index_norm = {}
    for i, pw in enumerate(generated_passwords):
        key = normalize_for_coverage(pw)
        if key not in generated_index_norm:
            generated_index_norm[key] = i + 1

    ranks = sorted(
        [
            generated_index_norm[normalize_for_coverage(pw)]
            for pw in test_set   
            if normalize_for_coverage(pw) in generated_index_norm
        ]
    )

    if ranks:
        print(f"Found in top 10: {sum(1 for r in ranks if r <= 10)}")
        print(f"Found in top 100: {sum(1 for r in ranks if r <= 100)}")
        print(f"Found in top 1,000: {sum(1 for r in ranks if r <= 1000)}")
        print(f"Found in top 10,000: {sum(1 for r in ranks if r <= 10000)}")
        print(f"Median rank: {np.median(ranks):.0f} | Mean rank: {np.mean(ranks):.0f}")

    return {
        "hit_rate_exact": hit_rate_exact,
        "hits_exact": len(hits_exact),
        "hit_rate_backslash_aware": hit_rate_norm,
        "hits_backslash_aware": hits_norm_count,
        "ranks": ranks,
    }


def protocol_2_distribution_analysis(generated_passwords, test_data):
    print("\n" + "=" * 70)
    print("PROTOCOLE 2: DISTRIBUTION ANALYSIS")
    print("=" * 70)

    def analyze(passwords, label):
        print(f"\n--- {label} ---")
        lengths = [len(pw) for pw in passwords]
        print(
            f"Length - Mean: {np.mean(lengths):.2f}, Std: {np.std(lengths):.2f}, "
            f"Min: {min(lengths)}, Max: {max(lengths)}"
        )
        n = len(passwords)
        print(f"With digits:    {sum(any(c.isdigit() for c in pw) for pw in passwords)/max(1, n)*100:.1f}%")
        print(f"With uppercase: {sum(any(c.isupper() for c in pw) for pw in passwords)/max(1, n)*100:.1f}%")
        print(f"With lowercase: {sum(any(c.islower() for c in pw) for pw in passwords)/max(1, n)*100:.1f}%")
        print(f"With special:   {sum(any(not c.isalnum() for c in pw) for pw in passwords)/max(1, n)*100:.1f}%")
        cc = Counter(c for pw in passwords for c in pw)
        print(f"Most common chars: {cc.most_common(10)}")
        return lengths

    gen_len = analyze(generated_passwords[: len(test_data)], "GENERATED")
    test_len = analyze(test_data, "TEST DATA")
    print(f"\nLength difference: {abs(np.mean(gen_len) - np.mean(test_len)):.2f}")


def read_password_lines(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n\r") for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Exécuter les protocoles d'évaluation sur un fichier de mots de passe généré (10k) contre eval.txt"
    )
    ap.add_argument("--eval", default="eval.txt", help="Chemin du fichier d'évaluation/référence")
    ap.add_argument("--generated", default="10k_decoder2.txt", help="Chemin du fichier de mots de passe générés")
    ap.add_argument("--name", default="Decoder2-10k", help="Étiquette affichée dans les rapports")
    ap.add_argument(
        "--protocol",
        choices=["coverage", "distribution", "both"],
        default="both",
        help="Quel(s) protocole(s) exécuter",
    )
    args = ap.parse_args()

    test_data = read_password_lines(args.eval)
    generated_passwords = read_password_lines(args.generated)

    if args.protocol in ("coverage", "both"):
        protocol_1_coverage_analysis(generated_passwords, test_data, name=args.name)

    if args.protocol in ("distribution", "both"):
        protocol_2_distribution_analysis(generated_passwords, test_data)


if __name__ == "__main__":
    main()
