import argparse
import numpy as np
import random

def positive_int(n_str):
    err = argparse.ArgumentTypeError(f'{n_str} is not a positive integer')

    if not n_str.isdigit():
        raise err

    val = int(n_str)

    if (val < 1):
        raise err

    return val

parser = argparse.ArgumentParser(description='Generates text given input probability matrix file')
parser.add_argument('infile', type=argparse.FileType('rb'))
parser.add_argument('outfile', type=argparse.FileType('w'))
parser.add_argument('--words', type=positive_int, default=10)

args = parser.parse_args()

def rand_weighted(weights):
    cumul_weights = np.cumsum(weights)

    r = random.random()

    for i, weight in enumerate(cumul_weights):
        if (r < weight):
            return i

    return None

def generate_word(prob_mat):
    prev_char_indexes = [0] * dimensions
    word = ''

    while True:
        weights = prob_mat[tuple(prev_char_indexes)]
        char_idx = rand_weighted(weights)
        if (char_idx == 1):
            break
        word += text_chars[char_idx - 2]

        prev_char_indexes.append(char_idx)
        prev_char_indexes.pop(0)

    return word

# Configuration

INFILE = args.infile
OUTFILE = args.outfile
WORDS = args.words

npz = np.load(INFILE)

prob_mat = npz['prob_mat']
dimensions = int(npz['config'][0])
text_chars = str(npz['config'][1])

# Generation phase

gen_words = '\n'.join(generate_word(prob_mat) for i in range(WORDS))

OUTFILE.write(gen_words)

INFILE.close()
OUTFILE.close()
