#!/usr/bin/env python3

"""
ICassigner.py

Conservative phylogeny-guided assignment of Acinetobacter baumannii
International Clones (ICs) using a core genome tree and metadata.

Rules:
- Known IC labels are never overwritten
- Unassigned (UA) isolates are assigned ONLY if:
    * >= MIN_SUPPORT labelled neighbours
    * >= MIN_PROP majority support for a single IC
- Otherwise, isolates remain UA
"""

import argparse
import pandas as pd
from collections import Counter
from ete3 import Tree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conservative IC assignment from a phylogenetic tree"
    )
    parser.add_argument("--tree", required=True,
                        help="Newick tree file (e.g. RAxML core genome tree)")
    parser.add_argument("--metadata", required=True,
                        help="Metadata CSV containing sample IDs and IC labels")
    parser.add_argument("--tip_col", default="sample_id",
                        help="Column name for tree tip/sample IDs")
    parser.add_argument("--ic_col", default="IC",
                        help="Column name for International Clone labels")
    parser.add_argument("--min_support", type=int, default=10,
                        help="Minimum number of labelled neighbours required")
    parser.add_argument("--min_prop", type=float, default=0.9,
                        help="Minimum majority proportion required")
    parser.add_argument("--output", default="metadata_with_conservative_IC.csv",
                        help="Output CSV filename")
    return parser.parse_args()


def infer_ic(tree, tip, ic_map, min_support, min_prop):
    node = tree & tip
    while node:
        tips = node.get_leaf_names()
        labels = [
            ic_map.get(t)
            for t in tips
            if ic_map.get(t) not in [None, "UA"]
        ]

        if len(labels) >= min_support:
            counts = Counter(labels)
            ic, n = counts.most_common(1)[0]
            prop = n / sum(counts.values())

            if prop >= min_prop:
                return ic, n, prop

        node = node.up

    return "UA", 0, 0.0


def main():
    args = parse_args()

    tree = Tree(args.tree, format=1)
    meta = pd.read_csv(args.metadata)

    ic_map = dict(zip(meta[args.tip_col], meta[args.ic_col]))

    inferred_ic = []
    support_n = []
    support_prop = []

    for tip in meta[args.tip_col]:
        original_ic = ic_map.get(tip)

        if original_ic == "UA":
            ic, n, prop = infer_ic(
                tree,
                tip,
                ic_map,
                args.min_support,
                args.min_prop
            )
        else:
            ic, n, prop = original_ic, None, None

        inferred_ic.append(ic)
        support_n.append(n)
        support_prop.append(prop)

    meta["IC_tree_conservative"] = inferred_ic
    meta["IC_tree_support_n"] = support_n
    meta["IC_tree_support_prop"] = support_prop

    meta.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
