import os, csv, json, sys

in_f = open(sys.argv[1], "r")

for line in in_f:
  parts = line.strip().split("\t")
  src, prefix = parts[0].split(" . ")
  src = f"{src} ."
  prefix = f"{prefix}: "
  jsonl = {"translation": {}}
  jsonl["translation"]["src"] = src
  jsonl["translation"]["prefix"] = prefix
  jsonl["translation"]["tgt"] = parts[1]
  print(json.dumps(jsonl))