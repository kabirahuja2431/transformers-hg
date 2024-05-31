# Multilingual Syntactic Transformations
This repository contains code for replicating the experiments of [Coloring the Blank Slate: Pre-training Imparts a Hierarchical Inductive Bias to Sequence-to-sequence Models](https://aclanthology.org/2022.findings-acl.106/), published in Findings of ACL 2022.

## Dependencies
Use `conda env create -f environment.yml` to replicate our conda environment.

We use the following:
- Python 3.8
- Torch 1.8.1
- Transformers 4.4.2

## Overview
* `data/question_have-havent_en`: English question formation data
* `data/question_have-can_withquest_de`: German question formation data
* `data/question_have-can_de`: zero-shot cross-lingual German question formation data
* `data/passiv_en_nps`: English passivization data
* `data/passiv_de_nps`: German passivization data
* `data/passiv_en-de_nps`: zero-shot cross-lingual German passivization data
* `models/run_seq2seq.py`: for fine-tuning and evaluating pre-trained models on syntactic transformations
* `models/metrics.py`: implementation of each accuracy and error analysis metric used in our experiments
* `models/pred_eval.py`: for defining the output metrics used when evaluating models' transformational abilities
* `models/plot_learning_curve.py`: for plotting paper-ready learning curves
* `scripts/finetune_{model}_question_have_{language}.sh`: question formation fine-tuning scripts
* `scripts/eval_{model}_question_have_{language}.sh`: question formation evaluation scripts
* `scripts/finetune_{model}_passivize_{language}.sh`: passivization fine-tuning scripts
* `scripts/eval_{model}_passivize_{language}.sh`: passivization evaluation scripts


## Data
We fine-tune pre-trained sequence-to-sequence models on English and German question formation and passivization. The fine-tuning data can be found in the `data/` directory. The fine-tuning data are artificially generated using a CFG-like grammar.

Each directory contains training, development, test, and generalization sets. The training set is the supervision provided to the models: these consist of declarative examples (i.e., identity transformations) marked with the `decl:` prefix, as well as transformation examples marked with either the `passiv:` or `quest:` prefix for passivization or question formation, respectively. The development set is used to measure convergence. The test set is used to measure a model's generalization on in-distrubution transformations.

The generalization sets are the evaluation sets used to measure a model's inductive biases. These consist of out-of-distribution syntactic transformations where linear positional heuristics do not result in correct syntactic transformations.

### Zero-shot Data
Our zero-shot cross-lingual fine-tuning data is different from the monolingual data. Here, the training sets consist of declaratives and transformation examples in English, and also declaratives in the target language. Target-language declaratives greatly improve zero-shot performance without providing explicit supervision as to how to perform the target-language transformation. Note that the training set here does NOT contain any target-language transformation examples.

When evaluating zero-shot generalization, there are two generalization sets. The first is where a phrase/clause is attached to objects (marked with an `o`); this measures performance on in-distribution transformations across languages. The second is where a phrase/clause is attached to subjects (marked with an `s`); this measures performance on out-of-distribution transformations across languages.

## Fine-tuning Scripts
We provide fine-tuning scripts in `scripts/`. The file format is `finetune_{model_name}_{transformation_type}_{language}`. This will produce a directory at `--output_dir` containing model checkpoints at every 500 iterations.

For mBART, source and target language IDs are required. Use the `--source_prefix` and `--target_prefix` arguments for monolingual data. For zero-shot or multilingual data, place the language IDs in each example (see the `*.lang_id.json` files in the zero-shot data directories) and use the `--prefix_from_file` argument.

To fine-tune a model with randomly initialized parameters, use the `--random_weights` argument.

## Evaluation Scripts
Our evaluation scripts are also provided in `scripts/`. The file format is `finetune_{model_name}_{transformation_type}_{language}`. Here, be sure to use both the `--do_learning_curve` and `--predict_with_generate` arguments. These scripts evaluate each checkpoint in `--output_dir`, writing accuracy and prediction files in each checkpoint directory. At the end, the script will also produce a learning curve figure in `--output_dir`.

### Metrics
For evaluation, we primarily focus on **sequence accuracy** and **first/second word accuracy**. Sequence accuracy (`exact_match`) measures how many transformations were performed exactly correctly, as measured by the reference and output transformations containing the same tokens in the same order. First word accuracy (`first_word`; used for question formation) measures how many transformations contain the correct auxiliary at the start of the sentence. Second word accuracy (`second_word`; used for passivization) measures how many transformations inverted the correct object noun.

We also use other metrics for error analysis. These include:
* `second_np`: In passivization, is the second NP (including any attached PPs) in a passive correct?
* `second_np_no_pp`: In passivization, is the second noun in a passive correct?
* `first_np_case_incorrect`: In German passivization, is the case of the first NP wrong?
* `prepose_first`: In question formation, is the first auxiliary fronted?

All of our metrics are implemented in `models/metrics.py`.


## Visualizations
Our paper-ready learning curves were created using `models/plot_learning_curve.py`. This must be run after the evaluation script, as it relies on the prediction files for each checkpoint. Here's an example:

```
python plot_learning_curve.py --checkpoint_dir <dir_containing_checkpoints> --gold_filename <gen_or_test_file> --metrics "exact_match,first_word"
```

## Non-pre-trained Baselines
In our paper, we compare pre-trained seq2seq models to non-pre-trained seq2seq models trained from scratch on syntactic transformations. For the non-pre-trained models, we used the [transductions repository](https://github.com/clay-lab/transductions), developed by the [CLAY Lab](http://clay.yale.edu) at Yale University.

## License
Our code is made available under an [MIT License](https://github.com/sebschu/multilingual-transformations/blob/main/LICENSE).

## Citation
If you use or modify the materials in this repository, please use the following citation:

```
@inproceedings{mueller-etal-2022-coloring,
    title = "Coloring the Blank Slate: Pre-training Imparts a Hierarchical Inductive Bias to Sequence-to-sequence Models",
    author = "Mueller, Aaron  and
      Frank, Robert  and
      Linzen, Tal  and
      Wang, Luheng  and
      Schuster, Sebastian",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.106",
    doi = "10.18653/v1/2022.findings-acl.106",
    pages = "1352--1368",
}
```
