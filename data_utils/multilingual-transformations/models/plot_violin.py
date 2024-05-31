import numpy as np
import os
import re
import sys
import glob
import pickle
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

from metrics import compute_metrics

"""
METRIC_COLORMAP = {
    "first_np": "#ff7f0e",
    "second_np": "#2ca02c",
    "second_np_no_pp": "#d62728"
}
"""

def main():
      argparser = argparse.ArgumentParser()
      
      argparser.add_argument("--checkpoint_dirs")
      argparser.add_argument("--gold_filename")
      argparser.add_argument("--metrics")
      argparser.add_argument("--move_legend", action="store_true")
      
      argparser.add_argument("--out_dir", default=".")    
      args = argparser.parse_args()

      if "," in args.checkpoint_dirs:
          checkpoint_dirs = args.checkpoint_dirs.split(",")
      else:
          checkpoint_dirs = [args.checkpoint_dirs]
      
      model_results = defaultdict(dict)
      metric_names = args.metrics.split(",")
      metrics_str = "-".join(metric_names)
      basename = os.path.basename(args.gold_filename).replace(".json", "")
      for checkpoint_dir in checkpoint_dirs:
        model_name = os.path.dirname(checkpoint_dir).split("-")[0]
        model_name = model_name.split("/")[-1]
        for path in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*", "")):
          pred_filename = os.path.join(path, basename + ".eval_preds_seq2seq.txt")
          it_res = re.match(".*checkpoint-([0-9]+)[/].*", path)
          it = int(it_res.group(1))
          if it < 1000:
            continue
          model_results[model_name][it] = compute_metrics(metric_names, pred_filename, args.gold_filename) 
   
      metric_rename = {}
      for metric in metric_names:
        if metric == "exact_match":
          this_metric = {"exact_match": "sequence"}
        else:
          this_metric = {metric: metric.replace("_", " ")}
        metric_rename = {**metric_rename, **this_metric}  # combine dicts

      # convert to DataFrame
      df = pd.DataFrame.from_dict({(i,j): model_results[i][j]
                                        for i in model_results.keys()
                                        for j in model_results[i].keys()},
                                    orient='index')
      df = df.reset_index()
      index_names = {"level_0":"Model", "level_1":"Checkpoint"}
      df.rename(columns={**index_names, **metric_rename},
              inplace=True)
      df = df.melt(id_vars=['Model', 'Checkpoint'],
                        value_vars=metric_rename.values(),
                        var_name='Metric', value_name='Accuracy')

      # plot violins to summarize distributions
      # ax = sns.violinplot(x='Model', y='Accuracy', hue='Metric', data=df,
      #                    cut=0.1)# , inner='quartile')
      # plot distribution as individual points
      ax = sns.stripplot(x='Model', y='Accuracy', hue='Metric', data=df, jitter=True, 
                         dodge=True, alpha=.3, zorder=1)
      # plot mean points
      red = ["#FF0000", "#FF0000"]
      sns.set_palette(red)
      sns.pointplot(x="Model", y="Accuracy", hue="Metric",
        data=df, dodge=.8 - .8 / 2,
        join=False,
        markers="d", ci=None)
      # add points over the violin
      handles, labels = ax.get_legend_handles_labels()
      if not args.move_legend:
        loc = "lower right"
      else:
        loc = "center right"
      plt.legend(handles[:2], labels[:2], loc=loc)
      plt.ylim([-0.05, 1.05])
      if "passiv_en_nps/" in args.gold_filename:
        title = "English Passivization"
      elif "passiv_de_nps/" in args.gold_filename:
        title = "German Passivization"
      elif "have-havent_en" in args.gold_filename:
        title = "English Question Formation"
      elif "have-can_withquest_de" in args.gold_filename:
        title = "German Question Formation"
      # zero-shot scores
      elif "passiv_en-de" in args.gold_filename:
        if "pp_o" in args.gold_filename:
          title = "Zero-shot German Passivization (PP on obj)"
        elif "pp_s" in args.gold_filename:
          title = "Zero-shot German Passivization (PP on subj)"
      elif "have-can_de" in args.gold_filename:
        if "rc_o" in args.gold_filename:
          title = "Zero-shot German Question Formation (RC on obj)"
        elif "rc_s" in args.gold_filename:
          title = "Zero-shot German Question Formation (RC on subj)"
      else:
        title = None
      if title is not None:
        plt.title(title)
      plt.savefig(os.path.join(args.out_dir, basename + "." + metrics_str + ".violin.png"), bbox_inches='tight')


if __name__ == '__main__':
  main()
