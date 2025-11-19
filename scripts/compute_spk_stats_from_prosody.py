#!/usr/bin/env python3
import argparse
import ast
import json
import numpy as np
from pathlib import Path


def load_entries(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            entries.append(ast.literal_eval(line))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std for pitch and energy in symbol-prosody tuples."
    )
    parser.add_argument("input_txt", type=Path, help="File produced by extract_symbol_prosody.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write stats to; prints to stdout when omitted.",
    )
    args = parser.parse_args()

    entries = load_entries(args.input_txt)
    if not entries:
        raise RuntimeError(f"No entries found in {args.input_txt}")

    pitch_vals, energy_vals = [], []
    for entry in entries:
        if len(entry) < 4:
            raise ValueError("Each tuple must contain symbols, durations, pitch, energy.")
        _, _, pitch, energy = entry
        # mimic training stats: only keep voiced pitch / non-zero energy
        pitch_vals.extend(float(x) for x in pitch if float(x) > 0.0)
        energy_vals.extend(float(x) for x in energy if float(x) > 0.0)

    if not pitch_vals or not energy_vals:
        raise RuntimeError("No non-zero pitch or energy values found; check the input file.")

    pitch_arr = np.array(pitch_vals, dtype=np.float64)
    energy_arr = np.array(energy_vals, dtype=np.float64)

    stats = {
        "pitch": {
            "mean": float(pitch_arr.mean()),
            "std": float(pitch_arr.std(ddof=0)),
        },
        "energy": {
            "mean": float(energy_arr.mean()),
            "std": float(energy_arr.std(ddof=0)),
        },
    }

    payload = json.dumps(stats, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
