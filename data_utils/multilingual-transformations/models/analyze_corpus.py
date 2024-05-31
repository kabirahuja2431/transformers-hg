import sys
import spacy
import argparse
from datasets import load_dataset
from tqdm import tqdm

QUESTION_AUXILIARIES = set(["have", "haven't", "has", "hasn't", "do", "don't", "does", "doesn't", "is", "isn't", "are", "aren't", "hat", "haben", "ist", "sind", "kann", "k\xf6nnen"])
JACCARD_THRESHOLD = 0.7     # totally arbitrary

# UTILITY FUNCTIONS
def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo and hi bound the slice of a to be searched.
    """
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x[1] > a[mid][1]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

# SIMILARITY METRICS
# calculate proportion of word set overlap across two sentences
# TODO: take into account words that appear multiple times (use Counter?)
def jaccard_sim(sent_1, sent_2):
    wordset_1 = set(sent_1)
    wordset_2 = set(sent_2)
    shared_words = wordset_1.intersection(wordset_2)
    jaccard_sim = len(shared_words) / \
            (len(wordset_1) + len(wordset_2) - len(shared_words))
    return jaccard_sim

def dependency_sim(sent_1, sent_2):
    pass


METRIC_FUNCTIONS = {
    "jaccard_sim": jaccard_sim,
    "dependency_sim": dependency_sim
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_examples", type=int, help="Number of examples to analyze.")
    parser.add_argument("-l", "--language", type=str, default="en",
                        help="Language on which we perform the corpus analysis.")
    parser.add_argument("-m", "--metric", type=str, default="jaccard_sim",
                        help="Similarity metric for comparing sentences.")
    parser.add_argument("-i", "--ignore-case", action="store_true",
                        help="Whether to ignore casing in input text.")
    parser.add_argument("-n", "--num-sents", type=int, default=50,
                        help="Display the top <num_sents> sentences by similarity.")
    parser.add_argument("-p", "--print-sents", action="store_true",
                        help="Print all sentences meeting extra criteria. Overrides --num_sents.")
    args = parser.parse_args()


    en_mc4 = load_dataset("mc4", args.language, split="train", streaming=True)
    # make it iterable using `next`
    en_mc4 = iter(en_mc4)

    if args.language == "en":
        # for this to work, you'll have to run `python -m spacy download en_core_web_sm`
        nlp = spacy.load("en_core_web_sm")
    else:
        raise NotImplementedError(f"Language \"{args.language}\" not supported.")

    skipped = 0
    max_sim_sents = []  # format of elements: ([sent_1, sent_2], similarity_score)
    # iterate through mC4 documents looking for disambiguating sentence pairs
    for _ in tqdm(range(args.num_examples)):
        example = next(en_mc4)["text"]
        if args.ignore_case:
            example = example.lower()
        doc = nlp(example)
        sentences = list(doc.sents)
        # skip document if only 1 sentence is present
        if len(sentences) < 2:
            skipped += 1
            continue

        # compare adjacent sentences in the training example
        prev_sent = None
        for idx in range(1, len(sentences)):
            if not prev_sent:
                sent_1 = [token.text for token in sentences[idx-1]]
            else:
                sent_1 = prev_sent
            # sentences are stored as lists of tokens (incl. punctuation)
            sent_2 = [token.text for token in sentences[idx]]
            prev_sent = sent_2

            sent_sim = METRIC_FUNCTIONS[args.metric](sent_1, sent_2)
            sent_tuple = ([sent_1, sent_2], sent_sim)

            if args.print_sents:
                # extra criteria
                if sent_sim < JACCARD_THRESHOLD:
                    continue
                aux_near_beginning = False
                for i in range(min(len(sent_1), len(sent_2), 3)):
                    if sent_1[i] in QUESTION_AUXILIARIES or sent_2[i] in QUESTION_AUXILIARIES:
                        aux_near_beginning = True
                if not aux_near_beginning:
                    continue
                # met all the criteria! print sentence
                print(f"{args.metric}: {sent_sim}")
                print(f"{' '.join(sent_1)}\n---\n{' '.join(sent_2)}")
                print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
            else:       # just print the k top sentences at the end
                if len(max_sim_sents) < args.num_sents:
                    reverse_insort(max_sim_sents, sent_tuple)
                elif sent_sim > max_sim_sents[-1][1]:
                    reverse_insort(max_sim_sents, sent_tuple)
                    max_sim_sents.pop()

    if not args.print_sents:
        max_sim_sents = reversed(max_sim_sents)
        for sent_tuple in max_sim_sents:
            sent_1, sent_2 = sent_tuple[0]
            sent_sim = sent_tuple[1]
            print(f"{args.metric}: {sent_sim}")
            print(f"\n{' '.join(sent_1)}\n---\n{' '.join(sent_2)}\n")
            print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
        print(f"# skipped documents: {skipped}")
