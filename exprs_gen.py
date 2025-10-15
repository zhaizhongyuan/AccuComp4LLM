#!/usr/bin/env python3
# SymbolicMathematics Expression Generator — Numbers-Only Leaves
# CSV output: infix,result (no header). Resample on NaN or infinite result.
from __future__ import annotations

import argparse
import re
import csv
import json
import os  # [NEW]
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict, Any  # [CHANGED]
import numpy as np
import math
from pathlib import Path  # [NEW]

# ------------------------------ Operators ------------------------------

OPERATORS = {
    # binary
    "add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2, "rac": 2, "derivative": 2,
    # unary
    "inv": 1, "pow2": 1, "pow3": 1, "pow4": 1, "pow5": 1, "sqrt": 1,
    "exp": 1, "ln": 1, "abs": 1, "sign": 1,
    # trig + inverse trig
    "sin": 1, "cos": 1, "tan": 1, "cot": 1, "sec": 1, "csc": 1,
    "asin": 1, "acos": 1, "atan": 1, "acot": 1, "asec": 1, "acsc": 1,
    # hyperbolic + inverse hyperbolic
    "sinh": 1, "cosh": 1, "tanh": 1, "coth": 1, "sech": 1, "csch": 1,
    "asinh": 1, "acosh": 1, "atanh": 1, "acoth": 1, "asech": 1, "acsch": 1,
    # helpers (kept for compatibility; not used specially here)
    "f": 1, "g": 2, "h": 3,
}

DEFAULT_OPERATORS = (
    "add:10,sub:3,mul:10,div:5,"
    "sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,"
    "ln:4,exp:4,"
    "sin:4,cos:4,tan:4,"
    "asin:1,acos:1,atan:1,"
    "sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"
)

# ------------------------------ Helpers ------------------------------

def parse_weighted_ops(spec: str) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    pairs = [x.strip().split(":") for x in spec.split(",") if x.strip()]
    for p in pairs:
        if len(p) != 2:
            raise ValueError(f"Bad operator spec fragment: {p}")
        if p[0] not in OPERATORS:
            raise ValueError(f"Unknown operator token '{p[0]}'")
        float(p[1])
    ops = [o for o, _ in pairs]
    probs = np.array([float(w) for _, w in pairs], dtype=np.float64)
    probs = probs / probs.sum()

    una_ops = [o for o in ops if OPERATORS[o] == 1]
    bin_ops = [o for o in ops if OPERATORS[o] == 2]
    una_probs = np.array([float(w) for o, w in pairs if OPERATORS[o] == 1], dtype=np.float64)
    bin_probs = np.array([float(w) for o, w in pairs if OPERATORS[o] == 2], dtype=np.float64)
    if len(una_ops) > 0:
        una_probs = una_probs / una_probs.sum()
    if len(bin_ops) > 0:
        bin_probs = bin_probs / bin_probs.sum()
    return ops, probs, una_ops, una_probs, bin_ops, bin_probs

def write_infix(token: str, args: Sequence[str]) -> str:
    if token == "add": return f"({args[0]})+({args[1]})"
    if token == "sub": return f"({args[0]})-({args[1]})"
    if token == "mul": return f"({args[0]})*({args[1]})"
    if token == "div": return f"({args[0]})/({args[1]})"
    if token == "pow": return f"({args[0]})**({args[1]})"
    if token == "rac": return f"({args[0]})**(1/({args[1]}))"
    if token == "abs": return f"Abs({args[0]})"
    if token == "inv": return f"1/({args[0]})"
    if token == "pow2": return f"({args[0]})**2"
    if token == "pow3": return f"({args[0]})**3"
    if token == "pow4": return f"({args[0]})**4"
    if token == "pow5": return f"({args[0]})**5"
    if token in [
        "sign","sqrt","exp","ln","sin","cos","tan","cot","sec","csc",
        "asin","acos","atan","acot","asec","acsc",
        "sinh","cosh","tanh","coth","sech","csch",
        "asinh","acosh","atanh","acoth","asech","acsch",
    ]:
        return f"{token}({args[0]})"
    if token == "derivative":
        return f"Derivative({args[0]},{args[1]})"
    if token == "f": return f"f({args[0]})"
    if token == "g": return f"g({args[0]},{args[1]})"
    if token == "h": return f"h({args[0]},{args[1]},{args[2]})"
    if token.startswith("INT"): return f"{token[-1]}{args[0]}"
    return token

def prefix_to_infix(prefix_tokens: Sequence[str], int_base: int = 10, balanced: bool = False) -> str:
    def is_digit_tok(t: str) -> bool:
        return re.fullmatch(r"-?\d+", t) is not None

    def parse_int(lst: List[str]) -> Tuple[str, List[str]]:
        if not lst: raise ValueError("Unexpected end while parsing INT")
        tag = lst[0]
        if not tag.startswith("INT"):
            raise ValueError("parse_int called on non-INT token")
        rest = lst[1:]
        digits: List[int] = []
        while rest and is_digit_tok(rest[0]):
            digits.append(int(rest[0]))
            rest = rest[1:]
        if not digits:
            return "0", rest
        if balanced:
            b = int_base; val = 0
            for d in digits: val = val*b + d
        else:
            if int_base > 0:
                b = int_base; val = 0
                for d in digits: val = val*b + d
                if tag == "INT-": val = -val
            else:
                b = int_base; val = 0
                for d in digits: val = val*b + d
        return str(val), rest

    def parse(lst: List[str]) -> Tuple[str, List[str]]:
        if not lst: raise ValueError("Empty prefix")
        t, rest = lst[0], lst[1:]
        if t in OPERATORS:
            n = OPERATORS[t]
            args = []
            for _ in range(n):
                arg, rest = parse(rest); args.append(arg)
            return write_infix(t, args), rest
        elif t.startswith("INT"):
            val, rest = parse_int([t] + rest); return val, rest
        else:
            return t, rest

    out, rem = parse(list(prefix_tokens))
    if rem: raise ValueError(f"Unparsed tail in prefix: {rem}")
    return f"({out})"

# ------------------------------ Tree construction [NEW] ------------------------------

@dataclass
class TreeNode:
    """A generic node in the generated expression tree."""
    token: str
    children: List["TreeNode"]

    # Fields used for integer leaves (when token starts with 'INT')
    is_int: bool = False
    int_tag: Optional[str] = None
    digits: Optional[List[int]] = None
    int_base: Optional[int] = None
    balanced: Optional[bool] = None
    int_value: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        if self.is_int:
            return {
                "type": "int",
                "tag": self.int_tag,
                "digits": self.digits,
                "base": self.int_base,
                "balanced": self.balanced,
                "value": self.int_value,
            }
        return {
            "type": "op",
            "token": self.token,
            "arity": len(self.children),
            "children": [c.to_json() for c in self.children],
        }

def prefix_to_tree(prefix_tokens: Sequence[str], int_base: int = 10, balanced: bool = False) -> TreeNode:
    """Turn the exact prefix tokens into a concrete tree structure (operators + INT leaves)."""
    toks = list(prefix_tokens)  # copy
    i = 0

    def is_digit_tok(t: str) -> bool:
        return re.fullmatch(r"-?\d+", t) is not None

    def parse_int_from(i0: int) -> Tuple[TreeNode, int]:
        if i0 >= len(toks):
            raise ValueError("Unexpected end while parsing INT")
        tag = toks[i0]
        if not tag.startswith("INT"):
            raise ValueError("parse_int_from called on non-INT token")
        i1 = i0 + 1
        digits: List[int] = []
        while i1 < len(toks) and is_digit_tok(toks[i1]):
            digits.append(int(toks[i1]))
            i1 += 1
        # compute value in the same way as prefix_to_infix
        if not digits:
            val = 0
        elif balanced:
            b = int_base; val = 0
            for d in digits:
                val = val * b + d
        else:
            if int_base > 0:
                b = int_base; val = 0
                for d in digits:
                    val = val * b + d
                if tag == "INT-":
                    val = -val
            else:
                b = int_base; val = 0
                for d in digits:
                    val = val * b + d
        node = TreeNode(
            token=tag, children=[],
            is_int=True, int_tag=tag, digits=digits,
            int_base=int_base, balanced=balanced, int_value=val
        )
        return node, i1

    def parse_from(i0: int) -> Tuple[TreeNode, int]:
        if i0 >= len(toks):
            raise ValueError("Empty prefix")
        t = toks[i0]
        if t in OPERATORS:
            n = OPERATORS[t]
            children: List[TreeNode] = []
            j = i0 + 1
            for _ in range(n):
                child, j = parse_from(j)
                children.append(child)
            return TreeNode(token=t, children=children), j
        elif t.startswith("INT"):
            return parse_int_from(i0)
        else:
            # Fallback: treat as symbol leaf (shouldn't appear in this generator)
            return TreeNode(token=t, children=[]), i0 + 1

    root, j = parse_from(0)
    if j != len(toks):
        raise ValueError(f"Unparsed tail in prefix: {toks[j:]}")
    return root

def tree_to_dot(root: TreeNode) -> str:
    """Convert a TreeNode into a Graphviz DOT graph as a string."""
    lines = ["digraph expr {", "  node [shape=box];"]
    counter = [0]
    def new_id() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def escape(s: str) -> str:
        return s.replace('"', '\\"')

    def label_for(node: TreeNode) -> str:
        if node.is_int:
            tag = node.int_tag or "INT"
            val = node.int_value if node.int_value is not None else "?"
            return f"{tag}\\n{val}"
        else:
            return node.token

    def walk(node: TreeNode) -> Tuple[str, List[str]]:
        me = new_id()
        lbl = label_for(node)
        decl = f'  {me} [label="{escape(lbl)}"];'
        edges: List[str] = []
        child_ids: List[str] = []
        for ch in node.children:
            cid, sublines = walk(ch)
            edges.extend(sublines)
            child_ids.append(cid)
        for cid in child_ids:
            edges.append(f"  {me} -> {cid};")
        return me, [decl] + edges

    _, all_lines = walk(root)
    lines.extend(all_lines)
    lines.append("}")
    return "\n".join(lines)

# ------------------------------ Safe numeric evaluator ------------------------------

def _safe_env():
    def cot(x): return 1.0 / math.tan(x)
    def sec(x): return 1.0 / math.cos(x)
    def csc(x): return 1.0 / math.sin(x)

    def acot(x): return math.atan(1.0 / x) if x != 0 else math.copysign(math.pi/2, x)
    def asec(x): return math.acos(1.0 / x)
    def acsc(x): return math.asin(1.0 / x)

    def sinh(x): return math.sinh(x)
    def cosh(x): return math.cosh(x)
    def tanh(x): return math.tanh(x)

    def coth(x): return cosh(x) / sinh(x)
    def sech(x): return 1.0 / cosh(x)
    def csch(x): return 1.0 / sinh(x)

    def asinh(x): return math.asinh(x)
    def acosh(x): return math.acosh(x)
    def atanh(x): return math.atanh(x)

    def acoth(x): return 0.5 * math.log((x + 1.0) / (x - 1.0))
    def asech(x): return math.acosh(1.0 / x)
    def acsch(x): return math.asinh(1.0 / x)

    def ln(x): return math.log(x)
    def Abs(x): return abs(x)
    def sign(x): return -1.0 if x < 0 else (1.0 if x > 0 else 0.0)
    def Derivative(x, y): return 0.0  # numbers-only leaves → derivative is zero

    return {
        "__builtins__": {},
        "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": sinh, "cosh": cosh, "tanh": tanh,
        "asinh": asinh, "acosh": acosh, "atanh": atanh,
        "sqrt": math.sqrt, "exp": math.exp, "ln": ln,
        "Abs": Abs, "sign": sign,
        "cot": cot, "sec": sec, "csc": csc,
        "acot": acot, "asec": asec, "acsc": acsc,
        "coth": coth, "sech": sech, "csch": csch,
        "acoth": acoth, "asech": asech, "acsch": acsch,
        "Derivative": Derivative,
        "f": lambda x: x, "g": lambda x, y: x + y, "h": lambda x, y, z: x + y + z,
    }

def evaluate_infix(expr: str) -> float:
    try:
        val = eval(expr, _safe_env(), {})
        return float(val)
    except Exception:
        return float("nan")

# ------------------------------ Generator ------------------------------

@dataclass
class GenConfig:
    max_ops: int = 15
    max_int: int = 5
    positive: bool = True
    int_base: int = 10
    balanced: bool = False
    operators: str = DEFAULT_OPERATORS
    mode: str = "ubi"
    seed: Optional[int] = None

class ExprGen:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        (self.all_ops, self.all_probs,
         self.una_ops, self.una_probs,
         self.bin_ops, self.bin_probs) = parse_weighted_ops(cfg.operators)
        self.p1 = 1 if len(self.una_ops) > 0 else 0
        self.p2 = 1 if len(self.bin_ops) > 0 else 0
        if cfg.mode not in ("ubi", "bin"):
            raise ValueError("--mode must be 'ubi' or 'bin'")
        self.bin_dist = self._generate_bin_dist(cfg.max_ops)
        self.ubi_dist = self._generate_ubi_dist(cfg.max_ops)

    def _generate_bin_dist(self, max_ops: int) -> List[List[int]]:
        catalans = [1]
        for i in range(1, 2*max_ops + 1):
            catalans.append((4*i - 2) * catalans[i - 1] // (i + 1))
        D: List[List[int]] = []
        for e in range(max_ops + 2):
            row = []
            for n in range(2*max_ops + 2):
                if e == 0:
                    row.append(0)
                elif e == 1:
                    row.append(catalans[n] if n < len(catalans) else 0)
                else:
                    v1 = D[e-1][n+1] if (e-1) < len(D) and (n+1) < len(D[e-1]) else 0
                    v2 = D[e-2][n+1] if (e-2) >= 0 and (e-2) < len(D) and (n+1) < len(D[e-2]) else 0
                    row.append(max(v1 - v2, 0))
            D.append(row)
        return D

    def _generate_ubi_dist(self, max_ops: int) -> List[List[int]]:
        L = 1; p1, p2 = self.p1, self.p2
        D: List[List[int]] = []
        D.append([0] + [L**i for i in range(1, 2*max_ops + 3)])
        for n in range(1, 2*max_ops + 3):
            s = [0]
            for e in range(1, 2*max_ops - n + 3):
                left = s[e-1]
                up1  = D[n-1][e] if e < len(D[n-1]) else 0
                up2  = D[n-1][e+1] if (e+1) < len(D[n-1]) else 0
                s.append(L * left + p1 * up1 + p2 * up2)
            D.append(s)
        D2: List[List[int]] = []
        maxlen = max(len(x) for x in D)
        for i in range(maxlen):
            D2.append([D[j][i] for j in range(len(D)) if i < len(D[j])])
        return D2

    def _sample_next_pos_ubi(self, nb_empty: int, nb_ops: int) -> Tuple[int, int]:
        assert nb_empty > 0 and nb_ops > 0
        probs = []
        for i in range(nb_empty):
            a = self.ubi_dist[nb_empty - i][nb_ops - 1] if (nb_ops - 1) < len(self.ubi_dist[0]) else 0
            probs.append(self.p1 * a)
        for i in range(nb_empty):
            a = self.ubi_dist[nb_empty - i + 1][nb_ops - 1] if (nb_ops - 1) < len(self.ubi_dist[0]) else 0
            probs.append(self.p2 * a)
        total = self.ubi_dist[nb_empty][nb_ops] if nb_ops < len(self.ubi_dist[0]) else 0
        if total == 0:
            return self._sample_next_pos_bin(nb_empty, nb_ops), 2
        probs = np.array([p / total for p in probs], dtype=np.float64)
        idx = int(self.rng.choice(2*nb_empty, p=probs))
        arity = 1 if idx < nb_empty else 2
        e = idx % nb_empty
        return e, arity

    def _sample_next_pos_bin(self, nb_empty: int, nb_ops: int) -> int:
        assert nb_empty > 0 and nb_ops > 0
        numer = []
        for i in range(nb_empty):
            v = self.bin_dist[nb_empty - i + 1][nb_ops - 1] if (nb_ops - 1) < len(self.bin_dist[0]) else 0
            numer.append(v)
        denom = self.bin_dist[nb_empty][nb_ops] if nb_ops < len(self.bin_dist[0]) else 0
        if denom == 0:
            return int(self.rng.integers(nb_empty))
        probs = np.array([n / denom for n in numer], dtype=np.float64)
        return int(self.rng.choice(nb_empty, p=probs))

    def _write_int_tokens(self, val: int) -> List[str]:
        base = self.cfg.int_base
        balanced = self.cfg.balanced
        if balanced:
            if base <= 2:
                raise ValueError("balanced base must be > 2")
            digits = []; x = val; b = base; m = (b - 1)//2
            while x != 0:
                r = (x + m) % b - m
                x = (x - r) // b
                digits.append(str(r))
            if not digits: digits = ["0"]
            return ["INT"] + digits[::-1]
        else:
            if abs(base) < 2:
                raise ValueError("abs(base) must be >= 2")
            neg = False; x = val
            if base > 0:
                neg = x < 0; x = -x if neg else x
            digits = []; b = base
            while True:
                rem = x % b; x = x // b
                if rem < 0 or rem >= abs(b):
                    rem -= b; x += 1
                digits.append(str(rem))
                if x == 0: break
            tag = "INT" if base < 0 else ("INT-" if neg else "INT+")
            return [tag] + digits[::-1]

    def _get_leaf(self) -> List[str]:
        c = int(self.rng.integers(1, self.cfg.max_int + 1))
        if not self.cfg.positive and int(self.rng.integers(2)) == 1:
            c = -c
        return self._write_int_tokens(c)

    def generate_prefix(self, nb_total_ops: int) -> List[str]:
        stack: List[Optional[str]] = [None]
        nb_empty = 1
        l_leaves = 0
        t_leaves = 1

        for ops_left in range(nb_total_ops, 0, -1):
            if self.cfg.mode == "bin":
                skipped = self._sample_next_pos_bin(nb_empty, ops_left); arity = 2
            else:
                skipped, arity = self._sample_next_pos_ubi(nb_empty, ops_left)
                if arity == 1 and len(self.una_ops) == 0:
                    skipped = self._sample_next_pos_bin(nb_empty, ops_left); arity = 2

            if arity == 1:
                op = str(self.rng.choice(self.una_ops, p=self.una_probs))
            else:
                op = str(self.rng.choice(self.bin_ops, p=self.bin_probs))

            nb_empty += OPERATORS[op] - 1 - skipped
            t_leaves += OPERATORS[op] - 1
            l_leaves += skipped

            pos_list = [i for i, v in enumerate(stack) if v is None]
            pos = pos_list[l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(OPERATORS[op])] + stack[pos+1:]

        leaves = [self._get_leaf() for _ in range(t_leaves)]
        self.rng.shuffle(leaves)
        for i in range(len(stack)-1, -1, -1):
            if stack[i] is None:
                stack = stack[:i] + leaves.pop() + stack[i+1:]
        assert len(leaves) == 0

        return [str(s) for s in stack]

    def generate_infix(self, nb_total_ops: int) -> str:
        prefix = self.generate_prefix(nb_total_ops)
        return prefix_to_infix(prefix, int_base=self.cfg.int_base, balanced=self.cfg.balanced)

# ------------------------------ CLI ------------------------------

def main():
    p = argparse.ArgumentParser(description="Numbers-only SymbolicMathematics expression generator (CSV: infix,result).")
    p.add_argument("--num", type=int, default=5, help="Number of expressions to write")
    p.add_argument("--ops", type=int, default=None, help="Exact operator nodes per expression")
    p.add_argument("--max_ops", type=int, default=15, help="Maximum operator nodes (used if --ops is not set)")
    p.add_argument("--sample_ops", action="store_true", help="Sample k ~ Uniform[1, max_ops] when --ops is not set")

    p.add_argument("--max_int", type=int, default=5, help="Max absolute integer value for leaves")
    p.add_argument("--positive", type=str, default="true", help="If false, allow negative integers")
    p.add_argument("--int_base", type=int, default=10, help="Integer base for INT tokenization (>=2 or <=-2)")
    p.add_argument("--balanced", type=str, default="false", help="Use balanced representation for integers")

    p.add_argument("--operators", type=str, default=DEFAULT_OPERATORS, help="Operator:weight list")
    p.add_argument("--mode", type=str, default="ubi", choices=["ubi","bin"], help="ubi (unary+binary) or bin (binary only)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    p.add_argument("--out_csv", type=str, default="expressions.csv", help="CSV output path")
    p.add_argument("--quiet", action="store_true", help="Do not print expressions to stdout")

    # ------------------------------ New outputs ------------------------------
    p.add_argument("--out_trees", type=str, default=None,
                   help="JSONL output for trees (default: <out_csv>.trees.jsonl)")  # [NEW]
    p.add_argument("--dot_dir", type=str, default=None,
                   help="If set, write a Graphviz .dot file per expression to this directory")  # [NEW]
    p.add_argument("--with_ids", action="store_true",
                   help="If set, prepend an idx column to the CSV (1-based)")  # [NEW]

    args = p.parse_args()

    cfg = GenConfig(
        max_ops=args.max_ops,
        max_int=args.max_int,
        positive=(args.positive.lower() == "true"),
        int_base=args.int_base,
        balanced=(args.balanced.lower() == "true"),
        operators=args.operators,
        mode=args.mode,
        seed=args.seed,
    )
    gen = ExprGen(cfg)
    rng = np.random.default_rng(args.seed)

    out_csv_path = Path(args.out_csv)
    if args.out_trees is None:
        out_trees_path = out_csv_path.with_suffix(out_csv_path.suffix + ".trees.jsonl")
    else:
        out_trees_path = Path(args.out_trees)

    if args.dot_dir is not None:
        Path(args.dot_dir).mkdir(parents=True, exist_ok=True)

    written = 0
    # [CHANGED] open both CSV and JSONL
    with open(out_csv_path, "w", newline="") as f_csv, open(out_trees_path, "w", encoding="utf-8") as f_jsonl:
        w_csv = csv.writer(f_csv)
        while written < args.num:
            if args.ops is not None:
                nb_ops = int(args.ops)
            elif args.sample_ops:
                nb_ops = int(rng.integers(1, args.max_ops + 1))
            else:
                nb_ops = int(args.max_ops)

            # [CHANGED] Generate prefix first so we can build the tree
            prefix = gen.generate_prefix(nb_total_ops=nb_ops)
            infix = prefix_to_infix(prefix, int_base=cfg.int_base, balanced=cfg.balanced)
            val = evaluate_infix(infix)

            # Resample on NaN OR infinite:
            if not math.isfinite(val):
                continue

            idx = written + 1  # 1-based for readability / CSV line-number alignment

            if not args.quiet:
                if args.with_ids:
                    print(f"{idx},{infix},{val}")
                else:
                    print(f"{infix},{val}")

            # CSV row (optionally with idx first)
            if args.with_ids:
                w_csv.writerow([idx, infix, f"{val:.12g}"])
            else:
                w_csv.writerow([infix, f"{val:.12g}"])

            # Build and write tree JSON line
            tree = prefix_to_tree(prefix, int_base=cfg.int_base, balanced=cfg.balanced)
            rec = {
                "idx": idx,
                "ops": nb_ops,
                "infix": infix,
                "result": float(f"{val:.12g}"),
                "prefix": prefix,
                "tree": tree.to_json(),
                "config": {
                    "int_base": cfg.int_base,
                    "balanced": cfg.balanced,
                    "positive": cfg.positive,
                    "max_int": cfg.max_int,
                    "mode": cfg.mode,
                }
            }
            f_jsonl.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

            # Optional DOT file
            if args.dot_dir is not None:
                dot_text = tree_to_dot(tree)
                dot_path = Path(args.dot_dir) / f"expr_{idx:06d}.dot"
                with open(dot_path, "w", encoding="utf-8") as f_dot:
                    f_dot.write(dot_text)

            written += 1

if __name__ == "__main__":
    main()