import argparse
import copy
import os
import json
import pickle
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from one_state_grammar import OneStateGrammar
from inside_outside_em import training, set_initial_probabilities
from inside_outside.CFG import CFG as CFG_zhou
from inside_outside.PCFG_EM import PCFG_EM


def prepare_inputs(grammar, is_regular=False):

    # Get the nonterminals
    non_terminals = copy.deepcopy(grammar.nonterminals)

    # Get the terminals
    terminals = copy.deepcopy(grammar.terminals)

    # Get the productions
    productions = copy.deepcopy(grammar.cfg)

    # Seperate out unary and binary productions
    unary_rules = []
    binary_rules = []

    for nt in productions:
        for rule in productions[nt]:

            if len(rule) == 1:
                unary_rules.append((nt, rule[0], 0.0))
            elif len(rule) == 2:
                if is_regular:
                    binary_rules.append((nt, rule[0].capitalize(), rule[1], 0.0))
                    if rule[0].capitalize() not in non_terminals:
                        non_terminals.append(rule[0].capitalize())
                    unary_rule = (rule[0].capitalize(), rule[0], 0.0)
                    if unary_rule not in unary_rules:
                        unary_rules.append(unary_rule)
                else:
                    binary_rules.append((nt, rule[0], rule[1], 0.0))

    # Get the initial probabilities
    unary_rules = set_initial_probabilities(unary_rules)
    binary_rules = set_initial_probabilities(binary_rules)

    return (
        unary_rules,
        binary_rules,
        non_terminals,
    )


def prepare_inputs_zhou(cfg, is_regular=False):
    unary_rules, binary_rules, non_terminals = prepare_inputs(
        cfg, is_regular=is_regular
    )
    all_rules = unary_rules + binary_rules

    productions = []

    for rule in all_rules:
        lhs = rule[0]
        rhs = " ".join(list(rule[1:-1]))
        productions.append((lhs, rhs))
    return productions


def train_inside_outside(cfg, sentences, num_iter=5):

    is_regular = cfg.grammar_type == "RLG"
    productions = prepare_inputs_zhou(cfg, is_regular=is_regular)
    cfg_zhou = CFG_zhou(productions)
    pcfg_em_zhou = PCFG_EM(sentences, cfg_zhou)
    pcfg_productions = pcfg_em_zhou.EM(iter_num=num_iter)
    cfg.set_new_production_probs(pcfg_productions)


class PCFG:

    def __init__(
        self,
        grammar_type="CFG-CNF",
        filename="",
        productions={},
        cfg=None,
        theta_resolution=1000,
    ):
        assert filename != "" or productions != [] or cfg is not None
        self.grammar_type = grammar_type
        self.cfg, self.pcfg = self.load_pcfg(filename, productions, cfg)
        self.start = "S"
        self.nonterminals = list(self.cfg.keys())
        self.terminals = set()
        self.theta_resolution = theta_resolution
        for rhs in self.cfg.values():
            for prod in rhs:
                for token in prod:
                    if token not in self.nonterminals:
                        self.terminals.add(token)
        self.gen2parse_tree = {}
        self.mle_done = False

    def dfs(self, start):
        if start in self.terminals:
            return [start.strip()], [[]]
        else:
            expansions = []
            parse_trees = []
            for production in self.cfg[start]:
                expansion = [""]
                parse_tree = [[f"{start} -> {' '.join(production).strip()}"]]
                for symbol in production:
                    expansions_curr, parse_trees_curr = self.dfs(symbol)
                    expansion = [
                        x + " " + y for x in expansion for y in expansions_curr
                    ]
                    parse_tree = [x + y for x in parse_tree for y in parse_trees_curr]
                expansions.extend(expansion)
                parse_trees.extend(parse_tree)
            return expansions, parse_trees

    def randomized_dfs(self, start, parse_tree=[]):
        if start in self.terminals:
            return start.strip(), parse_tree
        else:
            production_id = np.random.choice(range(len(self.cfg[start])))
            production = self.cfg[start][production_id]
            expansion = ""
            for symbol in production:
                dfs_out, parse_tree = self.randomized_dfs(symbol, parse_tree)
                expansion += " " + dfs_out

            parse_tree.append(f"{start} -> {' '.join(production).strip()}")

            return expansion.strip(), parse_tree

    def generate(self):
        gens, parse_trees = self.dfs("S")
        assert len(gens) == len(parse_trees)
        gens = [" ".join(gen.strip().split()) for gen in gens]
        if self.gen2parse_tree == {}:
            for gen, parse_tree in zip(gens, parse_trees):
                if gen not in self.gen2parse_tree:
                    self.gen2parse_tree[gen] = []
                self.gen2parse_tree[gen].append(parse_tree)
        return gens, parse_trees

    def monte_carlo_parsing(self, sentence, N=1000):
        parses = []
        for _ in range(N):
            generation, parse_tree = self.randomized_dfs("S", [])
            if generation == sentence:
                if parse_tree not in parses:
                    parses.append(parse_tree)

        return parses

    def parse_likelihood(self, parse):
        prob = 1
        for rule in parse:
            lhs, rhs = rule.split(" -> ")
            rhs = rhs.split()
            # prob *= 1 / len(self.cfg[lhs])
            prob *= self.pcfg.get(tuple([lhs] + rhs))
        return prob

    def sentence_likelihood(self, sentence, N=1000):

        if self.gen2parse_tree == {}:
            self.generate()

        if "quest" in sentence or "dot" in sentence:
            sentence = sentence.replace("quest", "").replace("dot", "").strip()
        # parses = self.monte_carlo_parsing(sentence, N)
        parses = self.gen2parse_tree.get(sentence, [])
        likelihood = 0
        for parse in parses:
            likelihood += self.parse_likelihood(parse)
        return likelihood

    def corpus_log_likelihood(self, corpus, N=1000, io_num_iter=5):
        if not self.mle_done and self.grammar_type != "Flat":
            train_inside_outside(self, corpus, num_iter=io_num_iter)
            self.mle_done = True
        # breakpoint()
        self.set_theta_com()
        # breakpoint()
        log_likelihood = 0
        for sentence in corpus:
            sent_ll = 0
            max_tries = 20
            num_tries = 0
            sent_ll = self.sentence_likelihood(sentence, N)
            while sent_ll == 0:
                # print("Trying again!")
                if num_tries == max_tries:
                    break
                sent_ll = self.sentence_likelihood(sentence, N)
                num_tries += 1

            if sent_ll == 0:
                # print(sentence)
                log_likelihood += -np.inf
            else:
                log_likelihood += np.log(sent_ll)
        return log_likelihood

    def set_theta_com(self):
        def get_theta(lhs):
            rhses = self.cfg[lhs]
            thetas = []
            for rhs in rhses:
                thetas.append(self.pcfg[tuple([lhs] + rhs)])
            return thetas

        def set_theta(lhs, thetas):

            for idx, rhs in enumerate(self.cfg[lhs]):
                self.pcfg[tuple([lhs] + rhs)] = thetas[idx]

        # breakpoint()
        for lhs in self.nonterminals:
            thetas = get_theta(lhs)
            com = self.com_on_discrete_simplex(self.theta_resolution, thetas)
            set_theta(lhs, com)

    def grammar_log_prob(
        self, geom_p=0.5, nt_geom_p=None, prod_geom_p=None, item_geom_p=None
    ):
        N_non_terminals = len(self.nonterminals)
        prob_N_nt = stats.geom.pmf(
            N_non_terminals, p=geom_p if nt_geom_p is None else nt_geom_p
        )
        if self.grammar_type != "Flat":
            log_prior = np.log(prob_N_nt)
        else:
            log_prior = 0
        # print(log_prior)
        for non_terminal in self.nonterminals:
            prob_num_prods = stats.geom.pmf(
                len(self.cfg[non_terminal]),
                p=geom_p if prod_geom_p is None else prod_geom_p,
            )
            log_prior += np.log(prob_num_prods)

            # Compute log p(\theta_k)
            # print(-1 * (len(self.cfg[non_terminal]) - 1) * np.log(self.theta_resolution))
            log_prior += (
                -1 * (len(self.cfg[non_terminal]) - 1) * np.log(self.theta_resolution)
            )

            for rhs in self.cfg[non_terminal]:
                if self.grammar_type in ["CFG-CNF", "RLG"]:
                    prob_num_items = 1 / 2
                else:
                    prob_num_items = stats.geom.pmf(
                        len(rhs), p=geom_p if item_geom_p is None else item_geom_p
                    )
                log_prior += np.log(prob_num_items)

                for idx, token in enumerate(rhs):
                    if self.grammar_type == "CFG-CNF":
                        if len(rhs) == 2:
                            prob_token = 1 / (
                                len(self.nonterminals) - 1
                            )  # Exclude S from the list of non-terminals
                        else:
                            prob_token = 1 / len(self.terminals)
                    elif self.grammar_type == "RLG":
                        if idx == 0:
                            prob_token = 1 / len(self.terminals)
                        else:
                            prob_token = 1 / (
                                len(self.nonterminals) - 1
                            )  # Exclude S from the list of non-terminals
                    elif self.grammar_type == "Flat":
                        prob_token = 1 / len(self.terminals)
                    else:
                        prob_token = 1 / (
                            len(self.terminals) + len(self.nonterminals) - 1
                        )

                    log_prior += np.log(prob_token)

        log_theta_res = np.floor(np.log10(self.theta_resolution)) + 1
        prob_log_theta_res = stats.geom.pmf(log_theta_res, p=geom_p)
        log_prior += np.log(prob_log_theta_res)

        return log_prior

    def log_posterior(
        self,
        corpus,
        geom_p=0.5,
        N=1000,
        nt_geom_p=None,
        prod_geom_p=None,
        item_geom_p=None,
        io_num_iter=5,
    ):
        return self.corpus_log_likelihood(
            corpus, N, io_num_iter=io_num_iter
        ) + self.grammar_log_prob(
            geom_p,
            nt_geom_p=nt_geom_p,
            prod_geom_p=prod_geom_p,
            item_geom_p=item_geom_p,
        )

    def set_new_production_probs(self, productions):
        for rule in self.pcfg:
            lhs = rule[0]
            rhs = list(rule[1:])
            io_rhs = rhs
            if self.grammar_type == "RLG":
                if len(rhs) == 2:
                    io_rhs = [rhs[0].capitalize(), rhs[1]]
            prob = productions[tuple([lhs] + io_rhs)]
            self.pcfg[tuple([lhs] + rhs)] = prob

    def tune_g(self, corpus, geom_p=0.5, io_num_iter=5):
        log_posteriors = []
        for g in [1, 10, 100, 1000, 10000, 100000]:
            self.mle_done = False
            self.theta_resolution = g
            log_posterior = self.log_posterior(corpus, geom_p, io_num_iter=io_num_iter)
            log_posteriors.append(log_posterior)

        max_idx = np.argmax(log_posteriors)
        max_posterior = log_posteriors[max_idx]
        self.theta_resolution = [1, 10, 100, 1000, 10000, 100000][max_idx]
        return log_posteriors, max_posterior, self.theta_resolution

    def load_pcfg(self, filename="", productions=[], cfg=None):
        assert filename != "" or productions != [] or cfg is not None
        if cfg is not None:
            return self.load_pcfg_from_cfg(cfg)
        elif filename != "":
            return self.load_pcfg_from_file(filename)
        else:
            return self.load_pcfg_from_productions(productions)

    @staticmethod
    def load_pcfg_from_cfg(cfg):
        pcfg = defaultdict(float)
        for lhs in cfg:
            for rhs in cfg[lhs]:
                pcfg[tuple([lhs] + rhs)] = (1.0) / len(cfg[lhs])
        return cfg, pcfg

    @staticmethod
    def load_pcfg_from_productions(productions):
        cfg = defaultdict(list)
        for rule, prob in productions.items():
            if prob != 0:
                lhs = rule[0]
                rhs = list(rule[1:])
                cfg[lhs].append((rhs))
        return cfg, productions

    @staticmethod
    def load_pcfg_from_file(filename):
        with open(filename) as f:
            lines = f.readlines()
            productions = []
            for line in lines:
                if line.strip() == "":
                    continue
                rule = line.strip().split("\t")
                lhs = rule[1]
                rhs = rule[2:]
                productions.append([lhs, rhs])

        cfg = defaultdict(list)
        pcfg = defaultdict(float)
        for lhs, rhs in productions:
            cfg[lhs].append(rhs)
            pcfg[tuple([lhs] + rhs)] = 1.0

        # #Normalize
        for lhs, rhses in cfg.items():
            for rhs in rhses:
                pcfg[tuple([lhs] + rhs)] = 1.0 / len(rhses)

        return cfg, pcfg

    @staticmethod
    def com_on_discrete_simplex(g, theta_k):
        def find_grid_vertices(y, g):
            # Check if the point y lies inside the m-1 simplex
            if not np.isclose(np.sum(y), 1.0):
                # print(np.sum(y))
                y = y / np.sum(y)
                # raise ValueError("Point y should lie inside the m-1 simplex.")
            # breakpoint()
            # Number of dimensions
            m = len(y)

            # Initialize grid indices
            grid_indices = [0] * m

            # Compute grid indices for each dimension
            for i in range(m):
                x_i = y[i] * g
                grid_indices[i] = min(int(np.floor(x_i)), g - 1)

            # Generate all vertices of the grid
            vertices = []
            for offset in np.ndindex((2,) * (m)):
                vertex_indices = np.array(grid_indices) + np.array(offset)
                vertex = vertex_indices / g
                if not np.allclose(np.sum(vertex), 1.0, atol=1e-3):
                    continue
                # vertex = np.append(vertex, 1.0 - np.sum(vertex))
                vertices.append(vertex)

            return vertices

        def check_is_uniform(theta_k):

            return np.allclose(np.array(theta_k), 1 / len(theta_k), 1e-5)

        if check_is_uniform(theta_k):
            return theta_k

        vertices = find_grid_vertices(theta_k, g)

        com = np.array(vertices)
        if len(com.shape) != 1:
            com = com.mean(axis=0)
        return com


def merge_step(production_dict, nt1, nt2):

    merged_nt = f"m({nt1},{nt2})"

    updated_dict = copy.deepcopy(production_dict)

    # Replace all instances of nt1 and nt2 with merged_nt in updated_dict
    for nt in updated_dict:
        for i, prod in enumerate(updated_dict[nt]):
            updated_dict[nt][i] = [
                merged_nt if x == nt1 or x == nt2 else x for x in prod
            ]

        # Remove duplicate productions as the result of merge
        prods_after_rem = []
        visited_prods = set()
        for prod in updated_dict[nt]:
            if tuple(prod) not in visited_prods:
                prods_after_rem.append(prod)
                visited_prods.add(tuple(prod))
        updated_dict[nt] = prods_after_rem

    # Update productions of nt1 and nt2 as the productions of merged_nt
    updated_dict[merged_nt] = updated_dict[nt1] + updated_dict[nt2]

    # Remove duplicate productions as the result of merge
    prods_after_rem = []
    visited_prods = set()
    for prod in updated_dict[merged_nt]:
        if tuple(prod) not in visited_prods:
            prods_after_rem.append(prod)
            visited_prods.add(tuple(prod))
    updated_dict[merged_nt] = prods_after_rem

    # Remove nt1 and nt2 from updated_dict
    del updated_dict[nt1]
    del updated_dict[nt2]

    return updated_dict, merged_nt


def score_merges(
    grammar,
    corpus,
    geom_p=0.5,
    nt_geom_p=None,
    prod_geom_p=None,
    item_geom_p=None,
    io_num_iter=2,
):
    best_grammar = None
    best_log_posterior = float("-inf")
    prod_dict = grammar.cfg
    pcfg_prod_dict = grammar.pcfg
    non_terminals = [nt for nt in grammar.nonterminals if nt != "S"]
    merge_dict = {}
    for i, nt1 in enumerate(non_terminals):
        for j, nt2 in enumerate(non_terminals):
            if i < j:
                merged_dict, merge_nt = merge_step(prod_dict, nt1, nt2)
                new_grammar = PCFG(grammar_type=grammar.grammar_type, cfg=merged_dict)
                # print(new_grammar.nonterminals)
                # print(new_grammar.terminals)
                # print(new_grammar.cfg)
                try:
                    log_posterior = new_grammar.log_posterior(
                        corpus,
                        geom_p=geom_p,
                        nt_geom_p=nt_geom_p,
                        prod_geom_p=prod_geom_p,
                        item_geom_p=item_geom_p,
                        io_num_iter=io_num_iter,
                    )
                except RecursionError:
                    # print("Weird!")
                    continue
                if log_posterior > best_log_posterior:
                    best_log_posterior = log_posterior
                    best_grammar = new_grammar
                merge_dict[merge_nt] = {
                    "grammar": new_grammar,
                    "log_posterior": log_posterior,
                    "nt1": nt1,
                    "nt2": nt2,
                }

    return merge_dict, best_grammar, best_log_posterior


def get_log_posterior(
    new_grammar,
    corpus,
    geom_p=0.5,
    nt_geom_p=None,
    prod_geom_p=None,
    item_geom_p=None,
    io_num_iter=2,
):
    try:
        log_posterior = new_grammar.log_posterior(
            corpus,
            geom_p=geom_p,
            nt_geom_p=nt_geom_p,
            prod_geom_p=prod_geom_p,
            item_geom_p=item_geom_p,
            io_num_iter=io_num_iter,
        )
    except RecursionError:
        log_posterior = float("-inf")
    return log_posterior


def score_merges_parallel(
    grammar,
    corpus,
    geom_p=0.5,
    nt_geom_p=None,
    prod_geom_p=None,
    item_geom_p=None,
    io_num_iter=2,
    num_processes=4,
):
    best_grammar = None
    best_log_posterior = float("-inf")
    prod_dict = grammar.cfg
    pcfg_prod_dict = grammar.pcfg
    non_terminals = [nt for nt in grammar.nonterminals if nt != "S"]
    merge_dict = {}
    merged_grammars = []
    for i, nt1 in enumerate(non_terminals):
        for j, nt2 in enumerate(non_terminals):
            if i < j:
                merged_dict, merge_nt = merge_step(prod_dict, nt1, nt2)
                new_grammar = PCFG(grammar_type=grammar.grammar_type, cfg=merged_dict)
                merged_grammars.append(new_grammar)
                # print(new_grammar.nonterminals)
                # print(new_grammar.terminals)
                # print(new_grammar.cfg)

    # Get the log posteriors for all the merged grammars parallely using multiprocessing

    with multiprocessing.Pool(num_processes) as pool:
        log_posteriors = pool.starmap(
            get_log_posterior,
            [
                (
                    new_grammar,
                    corpus,
                    geom_p,
                    nt_geom_p,
                    prod_geom_p,
                    item_geom_p,
                    io_num_iter,
                )
                for new_grammar in merged_grammars
            ],
        )

    # Get the best grammar

    max_posterior = max(log_posteriors)
    best_grammar = merged_grammars[log_posteriors.index(max_posterior)]

    # for i, nt1 in enumerate(non_terminals):
    #     for j, nt2 in enumerate(non_terminals):
    #         if i < j:
    #             merge_nt = f"m({nt1},{nt2})"
    #             log_posterior = log_posteriors.pop(0)
    #             new_grammar = merged_grammars.pop(0)
    #             if log_posterior > best_log_posterior:
    #                 best_log_posterior = log_posterior
    #                 best_grammar = new_grammar
    #             merge_dict[merge_nt] = {
    #                 "grammar": new_grammar,
    #                 "log_posterior": log_posterior,
    #                 "nt1": nt1,
    #                 "nt2": nt2,
    # }

    # try:
    #     log_posterior = new_grammar.log_posterior(
    #         corpus,
    #         geom_p=geom_p,
    #         nt_geom_p=nt_geom_p,
    #         prod_geom_p=prod_geom_p,
    #         item_geom_p=item_geom_p,
    #         io_num_iter=io_num_iter,
    #     )
    # except RecursionError:
    #     # print("Weird!")
    #     continue
    # if log_posterior > best_log_posterior:
    #     best_log_posterior = log_posterior
    #     best_grammar = new_grammar
    # merge_dict[merge_nt] = {
    #     "grammar": new_grammar,
    #     "log_posterior": log_posterior,
    #     "nt1": nt1,
    #     "nt2": nt2,
    # }

    return merge_dict, best_grammar, max_posterior


def bayesian_grammar_merging_greedy(
    grammar,
    corpus,
    geom_p=0.5,
    nt_geom_p=None,
    prod_geom_p=None,
    item_geom_p=None,
    io_num_iter=2,
    parallel=False,
    num_processes=4,
):
    log_posterior_init = grammar.log_posterior(
        corpus,
        geom_p=geom_p,
        nt_geom_p=nt_geom_p,
        prod_geom_p=prod_geom_p,
        item_geom_p=item_geom_p,
        io_num_iter=io_num_iter,
    )
    print(f"Initial log posterior: {log_posterior_init}")

    log_posterior_curr = log_posterior_init
    curr_grammar = copy.deepcopy(grammar)
    while True:
        if parallel:
            _, best_grammar, best_log_posterior = score_merges_parallel(
                curr_grammar,
                corpus,
                geom_p=geom_p,
                nt_geom_p=nt_geom_p,
                prod_geom_p=prod_geom_p,
                item_geom_p=item_geom_p,
                io_num_iter=io_num_iter,
                num_processes=num_processes,
            )
        else:
            _, best_grammar, best_log_posterior = score_merges(
                curr_grammar,
                corpus,
                geom_p=geom_p,
                nt_geom_p=nt_geom_p,
                prod_geom_p=prod_geom_p,
                item_geom_p=item_geom_p,
                io_num_iter=io_num_iter,
            )
        print(f"Best log posterior: {best_log_posterior}")
        print(
            f"Best log posterior recomputed: {best_grammar.log_posterior(corpus, geom_p=geom_p, nt_geom_p=nt_geom_p, prod_geom_p=prod_geom_p, item_geom_p=item_geom_p )}"
        )
        print(f"Current log posterior: {log_posterior_curr}")
        if best_log_posterior > log_posterior_curr:
            curr_grammar = best_grammar
            log_posterior_curr = best_log_posterior
            print(f"New best log posterior: {log_posterior_curr}")

        else:
            break

    return curr_grammar


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--g1_type", default="CFG-CNF")
    parser.add_argument("--g1_name", default="agreement_hr_v4")
    parser.add_argument("--g2_type", default="RLG")
    parser.add_argument("--g2_name", default="agreement_linear_v4")
    parser.add_argument(
        "--minimize", choices=["none", "g1", "g2", "all"], default="none"
    )
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--geom_p", type=float, default=0.5)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/bor",
        help="Directory to save results",
    )
    args = parser.parse_args()

    g1_type = args.g1_type
    g2_type = args.g2_type
    g1_name = args.g1_name
    g2_name = args.g2_name

    g1 = PCFG(g1_type, filename=os.path.join("cfgs", g1_name + ".gr"))
    g2 = PCFG(g2_type, filename=os.path.join("cfgs", g2_name + ".gr"))

    g1_gens, _ = g1.generate()
    g2_gens, _ = g2.generate()

    # Get common generations of the two grammars
    corpus = list(set(g1_gens).intersection(set(g2_gens)))
    print("Total number of g1 generations: ", len(g1_gens))
    print("Total number of g2 generations: ", len(g2_gens))
    print(f"Number of common generations: {len(corpus)}")

    flat_productions = {}
    for sent in corpus:
        flat_productions[tuple(["S"] + sent.split())] = 1 / len(corpus)

    flat_grammar = PCFG("Flat", productions=flat_productions)

    one_state_grammar = OneStateGrammar(g1.terminals)

    if args.minimize == "g1" or args.minimize == "all":

        if os.path.exists(f"cfgs/{args.g1_name}_min.pkl"):
            with open(f"cfgs/{args.g1_name}_min.pkl", "rb") as f:
                productions = pickle.load(f)
            g1 = PCFG(g1_type, productions=productions)

        else:
            g1 = bayesian_grammar_merging_greedy(
                g1, corpus, parallel=True, num_processes=os.cpu_count()
            )
            with open(f"cfgs/{args.g1_name}_min.pkl", "wb") as f:
                pickle.dump(g1.pcfg, f)

    if args.minimize == "g2" or args.minimize == "all":
        if os.path.exists(f"cfgs/{args.g2_name}_min.pkl"):
            with open(f"cfgs/{args.g2_name}_min.pkl", "rb") as f:
                productions = pickle.load(f)
            g2 = PCFG(g2_type, productions=productions)

        else:
            g2 = bayesian_grammar_merging_greedy(
                g2, corpus, parallel=True, num_processes=os.cpu_count()
            )
            with open(f"cfgs/{args.g2_name}_min.pkl", "wb") as f:
                pickle.dump(g2.pcfg, f)

    g1_log_prior = g1.grammar_log_prob(args.geom_p)
    g2_log_prior = g2.grammar_log_prob(args.geom_p)
    flat_grammar_log_prior = flat_grammar.grammar_log_prob(args.geom_p)
    one_state_grammar_log_prior = one_state_grammar.grammar_log_prob(args.geom_p)

    print("Log-prior probabilities:")
    print(f"Grammar 1: {g1_log_prior}")
    print(f"Grammar 2: {g2_log_prior}")
    print(f"Flat Grammar: {flat_grammar_log_prior}")
    print(f"One State Grammar: {one_state_grammar_log_prior}")
    print("*" * 50)

    # Compute log-likelihood of each grammar
    g1_log_likelihood = g1.corpus_log_likelihood(corpus)
    g2_log_likelihood = g2.corpus_log_likelihood(corpus)
    flat_grammar_log_likelihood = flat_grammar.corpus_log_likelihood(corpus)
    one_state_grammar_log_likelihood = one_state_grammar.corpus_log_likelihood(corpus)

    print("Log-likelihoods:")
    print(f"Grammar 1: {g1_log_likelihood}")
    print(f"Grammar 2: {g2_log_likelihood}")
    print(f"Flat Grammar: {flat_grammar_log_likelihood}")
    print(f"One State Grammar: {one_state_grammar_log_likelihood}")
    print("*" * 50)

    # Compute log-posterior of each grammar
    g1_log_posterior = g1_log_likelihood + g1_log_prior
    g2_log_posterior = g2_log_likelihood + g2_log_prior
    flat_grammar_log_posterior = flat_grammar_log_likelihood + flat_grammar_log_prior
    one_state_grammar_log_posterior = (
        one_state_grammar_log_likelihood + one_state_grammar_log_prior
    )

    print("Log-posteriors:")
    print(f"Grammar 1: {g1_log_posterior}")
    print(f"Grammar 2: {g2_log_posterior}")
    print(f"Flat Grammar: {flat_grammar_log_posterior}")
    print(f"One State Grammar: {one_state_grammar_log_posterior}")
    print("*" * 50)

    # Write results to a csv file with rows as grammars and columns as metrics
    df = pd.DataFrame(
        [
            [g1_log_prior, g1_log_likelihood, g1_log_posterior],
            [g2_log_prior, g2_log_likelihood, g2_log_posterior],
            [
                flat_grammar_log_prior,
                flat_grammar_log_likelihood,
                flat_grammar_log_posterior,
            ],
            [
                one_state_grammar_log_prior,
                one_state_grammar_log_likelihood,
                one_state_grammar_log_posterior,
            ],
        ],
        columns=["Log-prior", "Log-likelihood", "Log-posterior"],
        index=[
            f"{args.g1_name} ({args.g1_type})",
            f"{args.g2_name} ({args.g2_type})",
            "Flat Grammar",
            "One State Grammar",
        ],
    )

    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(
        f"{args.save_dir}/{args.g1_name}_{args.g2_name}_minimize_{args.minimize}.csv"
    )
