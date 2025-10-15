python exprs_gen.py --num 1000 --ops 8 --mode bin \
  --operators "add:1,sub:1,mul:1,div:1" \
  --out_csv data/1000_gsm_exprs.csv \
  --out_trees data/1000_gsm_exprs.csv.trees.jsonl --dot_dir data/dots && \
for f in data/dots/*.dot; do dot -Tpng "$f" -o "${f%.dot}.png"; done
