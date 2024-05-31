from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats


class CFG:
    def __init__(self, grammar_type="CFG-CNF", filename="", productions=[]):
        assert filename != "" or productions != []
        self.grammar_type = grammar_type
        self.cfg = self.load_cfg(filename, productions)
        self.start = "S"
        self.nonterminals = list(self.cfg.keys())
        self.terminals = set()
        for rhs in self.cfg.values():
            for prod in rhs:
                for token in prod:
                    if token not in self.nonterminals:
                        self.terminals.add(token)

    def dfs(self, start):
        if start in self.terminals:
            return [start.strip()]
        else:
            expansions = []
            for production in self.cfg[start]:
                expansion = [""]
                for symbol in production:
                    expansion = [
                        x + " " + y for x in expansion for y in self.dfs(symbol)
                    ]
                expansions.extend(expansion)
            return expansions

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
        gens = self.dfs("S")
        gens = [" ".join(gen.strip().split()) for gen in gens]
        return gens

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
            prob *= 1 / len(self.cfg[lhs])
        return prob

    def sentence_likelihood(self, sentence, N=1000):
        if "quest" in sentence or "dot" in sentence:
            sentence = sentence.replace("quest", "").replace("dot", "").strip()
        parses = self.monte_carlo_parsing(sentence, N)
        likelihood = 0
        for parse in parses:
            likelihood += self.parse_likelihood(parse)
        return likelihood

    def corpus_log_likelihood(self, corpus, N=1000):
        log_likelihood = 0
        for sentence in corpus:
            sent_ll = self.sentence_likelihood(sentence, N)
            if sent_ll == 0:
                print(sentence)
            log_likelihood += np.log(sent_ll)
        return log_likelihood

    def grammar_log_prob(self, geom_p=0.5):
        N_non_terminals = len(self.nonterminals)
        prob_N_nt = stats.geom.pmf(N_non_terminals, p=geom_p)
        log_prior = np.log(prob_N_nt)
        for non_terminal in self.nonterminals:
            prob_num_prods = stats.geom.pmf(len(self.cfg[non_terminal]), p=geom_p)
            log_prior += np.log(prob_num_prods)

            for rhs in self.cfg[non_terminal]:
                if self.grammar_type in ["CFG-CNF", "RLG"]:
                    prob_num_items = 1 / 2
                else:
                    prob_num_items = stats.geom.pmf(len(rhs), p=geom_p)
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
        return log_prior

    def load_cfg(self, filename="", productions=[]):
        assert filename != "" or productions != []
        if filename != "":
            return self.load_cfg_from_file(filename)
        else:
            return self.load_cfg_from_productions(productions)

    @staticmethod
    def load_cfg_from_productions(productions):
        cfg = defaultdict(list)
        for lhs, rhs in productions:
            cfg[lhs].append(rhs)
        return cfg

    @staticmethod
    def load_cfg_from_file(filename):
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
        for lhs, rhs in productions:
            cfg[lhs].append(rhs)
        return cfg
