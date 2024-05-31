import glob
import sys
import re

wildcard_path = sys.argv[1]

print("iteration,exact_match,first_word_accuracy")

for path in glob.glob(wildcard_path):
  it_res = re.match(".*checkpoint-([0-9]+)[/].*", path)
  it = it_res.group(1)
  with open(path, "r") as res_file:
    results = []
    '''
    # OLD BEHAVIOR
    for line in res_file:
      parts = line.strip().split(":")
      results.append(parts[1].strip())
    res_str = ",".join(results)
    print(f"{it},{res_str}")
    '''
    
    res_file.readline()
    line = res_file.readline()
    print(line.strip())
