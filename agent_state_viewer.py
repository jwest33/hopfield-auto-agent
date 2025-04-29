#!/usr/bin/env python3
"""view_agent_state.py – inspect a saved *agent_state.npz*
===========================================================
A tiny CLI helper for peeking inside the agent‑persistence file produced by
*main.py*.  By default the script shows a concise summary; pass ``--full`` to
print the full Hopfield weight matrices (may be large).

Usage
-----
$ python view_agent_state.py               # looks for ./agent_state.npz
$ python view_agent_state.py path/to/file  # inspect different file
$ python view_agent_state.py --full        # dump all arrays verbatim

"""
from __future__ import annotations
import argparse, os, sys, datetime as dt
import numpy as np
from textwrap import indent

# ───────────────────── helpers ─────────────────────

def fmt_bool(b: bool) -> str:
    return "yes" if b else "no"


def ts_to_str(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y‑%m‑%d %H:%M:%S")


def print_header(title: str):
    print("\n" + title)
    print("─" * len(title))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="View Hopfield agent state")
    parser.add_argument("file", nargs="?", default="hopfield-network/agent_state.npz",
                        help="path to the .npz file (default: agent_state.npz)")
    parser.add_argument("--full", action="store_true", help="dump full arrays")
    args = parser.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"Error: '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.file, allow_pickle=True)

    # ───── basic physiology & position ─────
    print_header("Physiology & Position")
    print(f"position : {data['pos'].tolist()}")
    print(f"energy   : {float(data['energy']):.1f}")
    print(f"hunger   : {int(data['hunger'])}")
    print(f"pain     : {int(data['pain'])}")
    print(f"carrying : {fmt_bool(bool(data['carrying']))}")
    print(f"stored   : {int(data['store'])} food items")

    # ───── memory layer 0 ─────
    mem0_M, mem0_t = data['mem0_M'], data['mem0_t']
    print_header("Memory L0 (observation) summary")
    print(f"vectors stored : {mem0_M.shape[0]} / dim {mem0_M.shape[1]}")
    if mem0_t.size:
        print(f"oldest         : {ts_to_str(float(mem0_t.min()))}")
        print(f"newest         : {ts_to_str(float(mem0_t.max()))}")

    # ───── memory layer 1 ─────
    mem1_M, mem1_t = data['mem1_M'], data['mem1_t']
    print_header("Memory L1 (sequence) summary")
    print(f"vectors stored : {mem1_M.shape[0]} / dim {mem1_M.shape[1]}")
    if mem1_t.size:
        print(f"oldest         : {ts_to_str(float(mem1_t.min()))}")
        print(f"newest         : {ts_to_str(float(mem1_t.max()))}")

    # ───── optional full dump ─────
    if args.full:
        print_header("Full array dump")
        for key in data.files:
            print(f"{key}:\n" + indent(str(data[key]), "  ") + "\n")


if __name__ == "__main__":
    main()
