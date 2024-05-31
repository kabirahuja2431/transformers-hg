import glob
import sys
import re

wildcard_path = sys.argv[1]


for path in glob.glob(wildcard_path):
  it_res = re.match(".*checkpoint-([0-9]+)[/].*", path)
  it = it_res.group(1)
  with open(path, "r") as res_file:
    results = []
    for line in res_file:
      parts = line.strip().split(":")
      results.append(parts[1].strip())
    res_str = ",".join(results)

  with open(path, "w") as out_f:
    print("iteration,exact_match,first_word_accuracy", file=out_f)
    print(f"{it},{res_str}",file=out_f)

    
