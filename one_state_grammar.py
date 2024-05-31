import argparse
import copy
import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

CFG_DIR = "cfgs/"
DATA_DIR = "data_utils/"

class OneStateGrammar:
    def __init__(self, terminals):
        self.terminals = terminals
        self.nonterminals = ["S"]
        self.start = "S"
        self.cfg = {"S": [[terminal] for terminal in self.terminals]}
        self.gen2parse_tree = {}

        self.g = 1000

    def fit_transition_probs(self, corpus):
        self.token2prob = defaultdict(float)
        total_count = 0
        for sentence in corpus:
            for token in sentence.split():
                self.token2prob[token] += 1
                total_count += 1
        for token in self.token2prob:
            self.token2prob[token] /= total_count

        self.theta_k = np.array(list(self.token2prob.values()))
        self.terminals = list(self.token2prob.keys())

    def corpus_log_likelihood(self, corpus):
        self.fit_transition_probs(corpus)
        ll = 0
        for sentence in corpus:
            ll += self.sentence_loglikelihood(sentence)
        return ll

    def sentence_loglikelihood(self, sentence):
        theta_com = self.com_on_discrete_simplex(self.g, list(self.token2prob.values()))
        token2theta_com = dict(zip(self.terminals, theta_com))
        ll = 0

        for token in sentence.split():
            ll += np.log(token2theta_com[token])
        return ll

    def grammar_log_prob(self, geom_p=0.5):
        log_prior = 0
        log_g = np.floor(np.log10(self.g)) + 1
        prob_log_g = stats.geom.pmf(log_g, p=geom_p)
        log_prior = np.log(prob_log_g)

        p_theta_k = 1 / (self.g) ** (len(self.terminals) - 1)
        log_prior += np.log(p_theta_k)

        return log_prior

    def log_posterior(self, corpus, geom_p=0.5):
        return self.corpus_log_likelihood(corpus) + self.grammar_log_prob(geom_p)

    def tune_g(self, corpus, geom_p=0.5):
        log_posteriors = []
        for g in [1, 100, 1000, 1000]:
            self.g = g
            log_posterior = self.log_posterior(corpus, geom_p)
            log_posteriors.append(log_posterior)

        max_idx = np.argmax(log_posteriors)
        max_posterior = log_posteriors[max_idx]
        self.g = [1, 100, 1000, 10000][max_idx]
        return max_posterior, self.g

    @staticmethod
    def com_on_discrete_simplex(g, theta_k):
        def find_grid_vertices(y, g):
            # Check if the point y lies inside the m-1 simplex
            if not np.isclose(np.sum(y), 1.0):
                raise ValueError("Point y should lie inside the m-1 simplex.")

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

        vertices = find_grid_vertices(theta_k, g)

        com = np.array(vertices)
        if len(com.shape) != 1:
            com = com.mean(axis=0)
        return com