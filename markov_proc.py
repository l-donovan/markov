import argparse
import numpy as np
import random
import sys

def positive_int(n_str):
    err = argparse.ArgumentTypeError(f'{n_str} is not a positive integer')

    if not n_str.isdigit():
        raise err

    val = int(n_str)

    if (val < 1):
        raise err

    return val

parser = argparse.ArgumentParser(description='Calculates probability matrix given input text file')
parser.add_argument('infile', type=argparse.FileType('r'))
parser.add_argument('outfile', type=argparse.FileType('wb'))
parser.add_argument('--dimensions', type=positive_int, default=4)

args = parser.parse_args()

# Configuration

INFILE = args.infile
OUTFILE = args.outfile
TEXT_CHARS = 'abcdefghijklmnopqrstuvwxyz'
DROP_CHARS = '\'"'
ESC_CHAR = '\0'
DIMENSIONS = args.dimensions

valid_chars = TEXT_CHARS + DROP_CHARS
width = len(TEXT_CHARS) + 2
size = width ** DIMENSIONS
prob_mat = np.zeros((width,) * (DIMENSIONS + 1))

# Preprocessing phase

text = INFILE.read().lower()
raw_words = ' '.join(text.split('\n')).split()
valid_words = [word + ESC_CHAR for word in raw_words if all([char in valid_chars for char in word])]

# Weighting phase

for word in valid_words:
    prev_char_indexes = [0] * DIMENSIONS

    for i in range(len(word)):
        if word[i] in DROP_CHARS:
            continue

        char_idx = TEXT_CHARS.find(word[i]) + 2
        prob_mat[tuple(prev_char_indexes)][char_idx] += 1

        prev_char_indexes.append(char_idx)
        prev_char_indexes.pop(0)

# Normalization phase

idx = [-1] + [0] * (DIMENSIONS - 1)

for c in range(width ** DIMENSIONS):
    idx[0] += 1
    i = 0

    while (idx[i] >= width and i < (DIMENSIONS - 1)):
        idx[i] = 0
        i += 1
        idx[i] += 1

    mag = prob_mat[tuple(idx)].sum()

    if (mag > 0):
        prob_mat[tuple(idx)] /= mag

config = np.array([DIMENSIONS, TEXT_CHARS])
np.savez_compressed(OUTFILE, prob_mat=prob_mat, config=config)

INFILE.close()
OUTFILE.close()
