# -*- coding: utf-8 -*-
"""
evaluate_1m.py
Évaluer 1M de mots de passe générés pour Decoder2 et GRU.

Exécute :
- les protocoles de couverture/distribution depuis evaluation_protocols_100k.py

Utilisation :
    python evaluate_1m.py --decoder 1000k_decoder2.txt --gru 1000k_gru.txt --eval eval.txt
"""

import argparse
from pathlib import Path

from evaluation_protocols_100k import (
    protocol_1_coverage_analysis,
    protocol_2_distribution_analysis,
    read_password_lines,
)


def assert_exists(path: str, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} file not found: {path}")


def run_protocols(generated_passwords, test_data, name: str, protocol: str):
    if protocol in ("coverage", "both"):
        protocol_1_coverage_analysis(generated_passwords, test_data, name=name)

    if protocol in ("distribution", "both"):
        protocol_2_distribution_analysis(generated_passwords, test_data)


def main():
    ap = argparse.ArgumentParser(description="Évaluer les fichiers générés à 1M : Decoder2 + GRU")
    ap.add_argument(
        "--decoder",
        "--transformer",
        dest="decoder",
        default="1000k_decoder2.txt",
        help="Fichier généré par Decoder2 (alias : --transformer)",
    )
    ap.add_argument("--gru", default="1000k_gru.txt", help="Fichier généré par GRU")
    ap.add_argument("--eval", default="eval.txt", help="Fichier d'évaluation/référence")
    ap.add_argument(
        "--protocol",
        choices=["coverage", "distribution", "both"],
        default="both",
        help="Quel(s) protocole(s) exécuter pour chaque modèle",
    )
    args = ap.parse_args()

    assert_exists(args.decoder, "Decoder2")
    assert_exists(args.gru, "GRU")
    assert_exists(args.eval, "Eval")

    print("=" * 62)
    print("EVALUATION 1M - DECODER2 + GRU")
    print("=" * 62)
    print(f"Decoder2 file:    {args.decoder}")
    print(f"GRU file:         {args.gru}")
    print(f"Eval file:        {args.eval}")
    print(f"protocol:         {args.protocol}")

    test_data = read_password_lines(args.eval)
    decoder_passwords = read_password_lines(args.decoder)
    gru_passwords = read_password_lines(args.gru)

    print("")
    print("[1/2] Protocols (coverage + distribution) - Decoder2")
    run_protocols(decoder_passwords, test_data, "Decoder2-1M", args.protocol)

    print("")
    print("[2/2] Protocols (coverage + distribution) - GRU")
    run_protocols(gru_passwords, test_data, "GRU-1M", args.protocol)

    print("")
    print("Done: full 1M evaluation completed.")


if __name__ == "__main__":
    main()
