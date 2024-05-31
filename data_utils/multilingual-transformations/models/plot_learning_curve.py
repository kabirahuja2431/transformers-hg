import numpy as np
import os
import re
import sys
import glob
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

from metrics import compute_metrics

METRIC_COLORMAP = {
    "exact_match": "#1f77b4",
    "first_np": "#ff7f0e",
    "second_np": "#2ca02c",
    "second_np_no_pp": "#d62728"
}

def main():
      argparser = argparse.ArgumentParser()
      
      argparser.add_argument("--checkpoint_dir")
      argparser.add_argument("--gold_filename")
      argparser.add_argument("--metrics")
      
      argparser.add_argument("--out_dir")    
      args = argparser.parse_args()
      
      if args.out_dir is None:
        args.out_dir = args.checkpoint_dir

      eval_results = {}
      metric_names = args.metrics.split(",")
      metrics_str = "-".join(metric_names)
      basename = os.path.basename(args.gold_filename).replace(".json", "")
      for path in glob.glob(os.path.join(args.checkpoint_dir, "checkpoint-*", "")):

          pred_filename = os.path.join(path, basename + ".eval_preds_seq2seq.txt")
          it_res = re.match(".*checkpoint-([0-9]+)[/].*", path)
          it = int(it_res.group(1))
          print(">>>", it)
          eval_results[it] = compute_metrics(metric_names, pred_filename, args.gold_filename) 
      
          
      for m in metric_names:
        its = sorted(eval_results.keys())
        vals = []
        for it in its:
          vals.append(eval_results[it][m])
        # if m in METRIC_COLORMAP:
        #   plt.plot(its,vals, label=m.replace("_", " "), color=METRIC_COLORMAP[m])
        # else:
        if m == "exact_match":
          m = "sequence"
        plt.plot(its,vals, label=m.replace("_", " "))
    
        
            
      plt.legend()
      plt.ylim([-0.05, 1.05])
      plt.xlabel("Tuning Iterations")
      plt.ylabel("Accuracy")
      title = None
      if "passiv_en_nps/" in args.gold_filename:
        title = "English Passivization"
      elif "passiv_de_nps/" in args.gold_filename:
        title = "German Passivization"
      elif "have-havent_en" in args.gold_filename:
        title = "English Question Formation"
      elif "have-can_withquest_de" in args.gold_filename:
        title = "German Question Formation"
      elif "passiv_en-de" in args.gold_filename:
        if "pp_o" in args.gold_filename:
          title = "Zero-shot German Passivization (PP on obj)"
        elif "pp_s" in args.gold_filename:
          title = "Zero-shot German Passivization (PP on subj)"
        elif "test" in args.gold_filename:
          title = "Zero-shot German Passivization (test)"
      elif "have-can_de" in args.gold_filename:
        if "rc_o" in args.gold_filename:
          title = "Zero-shot German Question Formation (RC on obj)"
        elif "rc_s" in args.gold_filename:
          title = "Zero-shot German Question Formation (RC on subj)"
        elif "test" in args.gold_filename:
          title = "Zero-shot German Question Formation (test)"
      else:
        title = None
      if title is not None:
        plt.title(title)
      plt.savefig(os.path.join(args.out_dir, basename + "." + metrics_str + ".learning_curve.png"), bbox_inches='tight')



if __name__ == '__main__':
  main()
