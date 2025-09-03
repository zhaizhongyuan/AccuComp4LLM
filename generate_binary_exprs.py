#!/usr/bin/env python3
import csv
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

Number = Union[int, float]

# -------------------- Node --------------------
@dataclass
class Node:
    op: Optional[str] = None     # '+', '-', '*', '/', 'power' OR None for leaf
    value: Optional[int] = None  # integer leaves
    left: Optional["Node"] = None
    right: Optional["Node"] = None

BINARY_OPS = ['+', '-', '*', '/', 'power']

DEFAULT_OP_WEIGHTS = {
    '+': 3, '-': 3, '*': 3, '/': 2, 'power': 2
}

# -------------------- Config --------------------
@dataclass
class Config:
    int_low: int = -9
    int_high: int = 9
    op_weights: Dict[str, int] = None
    result_abs_limit: float = 1e12
    pow_exp_limit: int = 6     # clamp |exponent|
    denom_eps: float = 1e-12   # avoid division by near-zero

    def __post_init__(self):
        if self.op_weights is None:
            self.op_weights = DEFAULT_OP_WEIGHTS.copy()

# -------------------- Helpers --------------------
def _weighted_choice(rng: random.Random, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]

def random_leaf(cfg: Config, rng: random.Random) -> Node:
    lo, hi = cfg.int_low, cfg.int_high
    if lo > hi:
        raise ValueError("int_low must be <= int_high")
    return Node(value=rng.randint(lo, hi))

def to_string(node: Node) -> str:
    if node.op is None:
        return str(node.value)
    if node.op in ['+', '-', '*', '/']:
        return f"({to_string(node.left)} {node.op} {to_string(node.right)})"
    if node.op == 'power':
        return f"power({to_string(node.left)}, {to_string(node.right)})"
    raise ValueError(f"Unknown op {node.op}")

def eval_tree(node: Node) -> float:
    if node.op is None:
        return float(node.value)
    a = eval_tree(node.left)
    b = eval_tree(node.right)
    if node.op == '+':     return a + b
    if node.op == '-':     return a - b
    if node.op == '*':     return a * b
    if node.op == '/':     return a / b
    if node.op == 'power': return math.pow(a, b)
    raise ValueError(f"Unknown op {node.op}")

def _finite(x: float) -> bool:
    return math.isfinite(x)

def _safe_eval(cfg: Config, node: Node) -> Optional[float]:
    try:
        y = eval_tree(node)
        if _finite(y) and abs(y) <= cfg.result_abs_limit:
            return y
    except Exception:
        pass
    return None

def _random_op(cfg: Config, rng: random.Random) -> str:
    ops = BINARY_OPS
    wts = [cfg.op_weights.get(op, 1) for op in ops]
    return _weighted_choice(rng, ops, wts)

# -------------------- Shape & Counting --------------------
def _shape_full_binary_tree(internal_nodes: int, rng: random.Random) -> Node:
    """
    Build a random FULL binary tree shape with exactly `internal_nodes` internal nodes.
    Leaves get placeholder values; operators are assigned later.
    """
    if internal_nodes == 0:
        return Node(op=None, value=0)
    left_internal = rng.randint(0, internal_nodes - 1)
    right_internal = (internal_nodes - 1) - left_internal
    return Node(
        op="__PLACEHOLDER_OP__",
        left=_shape_full_binary_tree(left_internal, rng),
        right=_shape_full_binary_tree(right_internal, rng)
    )

def _count_internal(node: Node) -> int:
    if node.op is None:
        return 0
    return 1 + _count_internal(node.left) + _count_internal(node.right)

def _gather_leaves(node: Node, acc):
    if node.op is None:
        acc.append(node)
    else:
        _gather_leaves(node.left, acc)
        _gather_leaves(node.right, acc)

# -------------------- Filling & Assignment --------------------
def _fill_leaves_with_ints(node: Node, cfg: Config, rng: random.Random):
    if node.op is None:
        node.value = rng.randint(cfg.int_low, cfg.int_high)
        return
    _fill_leaves_with_ints(node.left, cfg, rng)
    _fill_leaves_with_ints(node.right, cfg, rng)

def _assign_ops(node: Node, cfg: Config, rng: random.Random):
    if node.op is None:
        return
    node.op = _random_op(cfg, rng)
    _assign_ops(node.left, cfg, rng)
    _assign_ops(node.right, cfg, rng)

# -------------------- Non-collapsing Repairs --------------------
def _nudge_random_leaf(node: Node, cfg: Config, rng: random.Random, avoid_zero: bool = False):
    leaves = []
    _gather_leaves(node, leaves)
    if not leaves:
        return
    choice = rng.choice(leaves)
    for _ in range(30):
        v = rng.randint(cfg.int_low, cfg.int_high)
        if avoid_zero and v == 0:
            continue
        choice.value = v
        break

def _steer_subtree_value_soft(node: Node, cfg: Config, rng: random.Random, target_small_int: int = 1):
    """
    Preserve SHAPE. Make subtree evaluate to a small bounded integer by
    switching ops to '+' locally and nudging leaves to small integers.
    """
    if node.op is None:
        lo = max(-abs(target_small_int), cfg.int_low)
        hi = min(abs(target_small_int), cfg.int_high)
        if lo > hi:
            lo, hi = cfg.int_low, cfg.int_high
        node.value = rng.randint(lo, hi)
        return
    node.op = '+'
    _steer_subtree_value_soft(node.left, cfg, rng, target_small_int)
    _steer_subtree_value_soft(node.right, cfg, rng, 0)

def _repair_for_validity(node: Node, cfg: Config, rng: random.Random, attempts_per_fix: int = 30) -> bool:
    """
    Ensure evaluability without collapsing structure.
    """
    if node.op is None:
        return True

    if not _repair_for_validity(node.left, cfg, rng, attempts_per_fix):
        return False
    if not _repair_for_validity(node.right, cfg, rng, attempts_per_fix):
        return False

    if node.op == '/':
        for _ in range(attempts_per_fix):
            denom = _safe_eval(cfg, node.right)
            if denom is None or abs(denom) < cfg.denom_eps:
                _nudge_random_leaf(node.right, cfg, rng, avoid_zero=True)
                continue
            break

    if node.op == 'power':
        for _ in range(attempts_per_fix):
            val_left  = _safe_eval(cfg, node.left)
            val_right = _safe_eval(cfg, node.right)
            if val_left is None or val_right is None:
                _nudge_random_leaf(node.left, cfg, rng)
                _nudge_random_leaf(node.right, cfg, rng)
                continue

            exp_rounded = int(round(val_right))

            # Make exponent small integer within bounds; preserve shape
            if abs(val_right - exp_rounded) > 1e-9 or abs(exp_rounded) > cfg.pow_exp_limit:
                target_exp = max(-cfg.pow_exp_limit, min(cfg.pow_exp_limit, exp_rounded or 1))
                _steer_subtree_value_soft(node.right, cfg, rng, target_small_int=target_exp)
                continue

            # base = 0 with non-positive exponent is invalid; nudge base away from 0
            if abs(val_left) < cfg.denom_eps and exp_rounded <= 0:
                _nudge_random_leaf(node.left, cfg, rng, avoid_zero=True)
                continue

            # looks acceptable
            break

    # final safety check on this subtree
    return _safe_eval(cfg, node) is not None

# -------------------- Public API --------------------
def generate_expression_by_internal_nodes(
    internal_nodes: int,
    rng: random.Random,
    cfg: Config,
    max_global_attempts: int = 400
) -> Tuple[str, float, Node]:
    """
    Build a random full binary tree with exactly `internal_nodes` internal nodes.
    Assign operators/leaves, repair for validity (shape-preserving), and return (expr_str, value, root).
    """
    if internal_nodes < 0:
        raise ValueError("internal_nodes must be >= 0")

    for _ in range(max_global_attempts):
        root = _shape_full_binary_tree(internal_nodes, rng)
        _fill_leaves_with_ints(root, cfg, rng)
        if internal_nodes > 0:
            _assign_ops(root, cfg, rng)

        ok = _repair_for_validity(root, cfg, rng)
        if not ok:
            continue

        # Invariant: exact internal-node count
        assert _count_internal(root) == internal_nodes, "Internal node count changed during repair!"

        val = _safe_eval(cfg, root)
        if val is None:
            continue

        return to_string(root), val, root

    raise RuntimeError("Failed to generate a valid expression after many attempts. "
                       "Try adjusting ranges/limits or changing the seed.")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Generate full-binary arithmetic expressions with a fixed number of internal nodes (shape preserved).")
    ap.add_argument("--count", type=int, default=1000, help="Number of expressions to generate.")
    ap.add_argument("--internal-nodes", type=int, default=5, help="Exact number of internal (operator) nodes.")
    ap.add_argument("--seed", type=int, default=20250827, help="Base RNG seed.")
    ap.add_argument("--outfile", type=str, default="expressions_by_nodes.csv", help="Output CSV path.")
    ap.add_argument("--int-low", type=int, default=-9, help="Minimum integer leaf value.")
    ap.add_argument("--int-high", type=int, default=9, help="Maximum integer leaf value.")
    ap.add_argument("--pow-exp-limit", type=int, default=6, help="Max |exponent| for power(a, b).")
    ap.add_argument("--abs-limit", type=float, default=1e12, help="Reject expressions with |value| above this.")
    ap.add_argument("--float-format", type=str, default="%.12g", help="Numeric formatting for CSV.")
    ap.add_argument("--with-header", action="store_true", help="Write a header row.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = Config(
        int_low=args.int_low,
        int_high=args.int_high,
        result_abs_limit=args.abs_limit,
        pow_exp_limit=args.pow_exp_limit,
    )

    rows = []
    i = 0
    while len(rows) < args.count:
        # deterministic but distinct samples
        local_rng = random.Random(args.seed + i)
        expr, val, _ = generate_expression_by_internal_nodes(
            internal_nodes=args.internal_nodes,
            rng=local_rng,
            cfg=cfg
        )
        rows.append((expr, val))
        i += 1

    with open(args.outfile, "w", newline="") as f:
        writer = csv.writer(f)
        if args.with_header:
            writer.writerow(["expression", "value"])
        for expr, val in rows:
            writer.writerow([expr, (args.float_format % val)])

    print(f"Wrote {len(rows)} expressions to {args.outfile}. "
          f"Each expression has exactly {args.internal_nodes} internal nodes and {args.internal_nodes + 1} leaves.")

if __name__ == "__main__":
    main()